# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb

def modified_euler_integrate( d_states_d_t, init_state, times ):
    init_state = tf.verify_tensor_all_finite(init_state, "init_state NOT finite")
    x = [init_state]
    h = times[1]-times[0]
    F = [] 
    for t2,t1 in zip(times[1:],times[:-1]):
        
        f1 = d_states_d_t( x[-1], t1 )
        f2 = d_states_d_t( x[-1] + h*f1, t2 )

        # TODO: 
        x.append(  x[-1] + 0.5*h*(f1+f2) )
        F.append(0.5*h*(f1+f2))
    
    return tf.stack(x, axis = -1 ),tf.stack(F, axis = -1 )

def gen_odeint_while_body(d_states_d_t, h, times, algorithm='modeuler'):
    if algorithm is 'modeuler':
        def modified_euler_while_body( idx, x ):

            t1 = times[idx]
            t2 = times[idx+1]
            xi = x[idx]
            f1 = d_states_d_t( xi, t1 ) #d_states_d_t看来是一个neural network, input-->output(a transfer function)
            f2 = d_states_d_t( xi + h*f1, t2 ) #t1,t2 are unused
            y = tf.expand_dims( xi+0.5*h*(f1+f2), 0) #把新扩展的轴作为tensor的第一维（原先的axis 1,2,3顺次变为axis 2,3,4）（1,batch_size,n_iwae,10） Heun method
            
            return [tf.add(idx,1), tf.concat([x,y],axis=0)] #（2,batch_size,n_iwae,10）-->(86,batch_size,n_iwae,10)
        return modified_euler_while_body
    elif algorithm is 'rk4':
        def rk4_while_body( idx, x ):

            t1 = times[idx]
            t2 = times[idx+1]
            xi = x[idx]
            k1 = h * d_states_d_t (xi, t1)
            k2 = h * (d_states_d_t (xi + k1 * 0.5, t1 + h * 0.5))
            k3 = h * (d_states_d_t (xi + k2 * 0.5, t1 + h * 0.5))
            k4 = h * (d_states_d_t (xi + k3, t1 + h))
            y = tf.expand_dims(xi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0)
            
            return [tf.add(idx,1), tf.concat([x,y],axis=0)]
        return rk4_while_body
    else:
        raise Exception("Unknown ODE integration algorithm: " + algorithm)

def gen_odeint_while_conditions(T):
    def odeint_while_conditions( idx, x ):
        return tf.less( idx+1, T )
    return odeint_while_conditions #return True/False

def integrate_while( d_states_d_t, init_state, times, algorithm='modeuler' ):
    init_state = tf.verify_tensor_all_finite(init_state, "init_state NOT finite")
    x = tf.expand_dims( init_state, 0)  #[init_state]
    h = times[1]-times[0] #delta_t
    T = len(times)
    F = [] 

    tf_times = tf.convert_to_tensor(times, np.float32)

    i0 = tf.Variable(0, trainable=False)
    n_species = x.shape[-1]
    shape_invariants = [tf.TensorShape(None), tf.TensorShape([None,None,None,n_species])] #tf.TensorShape([None,1,1,times.shape[0]])
    loop_vars = [i0, x]
    
    results = tf.while_loop( gen_odeint_while_conditions(T), \
                             gen_odeint_while_body(d_states_d_t, h, tf_times, algorithm), \
                             loop_vars, \
                             shape_invariants=shape_invariants )#loop_vars：每次循环要增补的变量列表其中

    x = results[-1] #取出tensor （86,batch_size,i_iwae,10）
    
    return x, None #tf.stack(x, axis = -1 ), None #tf.stack(F, axis = -1 )
