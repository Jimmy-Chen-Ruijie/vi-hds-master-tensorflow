# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import numpy as np
import tensorflow as tf
import os
import subprocess
import sys
import tempfile
import re

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/vi-hds
#   pytest tests

#from models import dr_constant
from src import utils
from src import procdata
from src import distributions
from src.run_xval import Runner, create_parser

# Load a spec (YAML)
parser = create_parser(False)
args = parser.parse_args(['./specs/dr_constant_icml.yaml'])
print(args.yaml)
print(type(args.yaml))
spec = utils.load_config_file(args.yaml)  # spec is a dict of dicts of dicts

para_settings = utils.apply_defaults(spec['params']) #返回值为字典：有则改之，无则加冕，字典spec['params']中如果有params没有的项，则添加到params的键值对中
data_settings = procdata.apply_defaults(spec["data"]) #返回值为字典：添加或者修改default的值
model = para_settings['model']

# Load the parameter priors
shared = dict([(k, np.exp(v['mu'])) for k, v in para_settings['shared'].items()]) #v是字典
priors = dict()
priors.update(para_settings['global']) #把括号中字典的键值对更新到priors中去
priors.update(para_settings['global_conditioned'])
priors.update(para_settings['local'])

# Define a parameter sample that is the mode of each LogNormal prior.   mode--众数
theta = distributions.DotOperatorSamples()
for k, v in priors.items():
    if k != "conditioning":
        if 'mu' in v: #键为'mu'是否在字典v中
            sample_value = np.exp(v['mu'])         
        else: 
            sample_value = shared[v['distribution']]
        theta.add(k, np.tile(sample_value, [1,1]).astype(np.float32)) #theta是一个字典：key->distribution name of each variable

# Add the constants separately
for k, v in para_settings['constant'].items():
    theta.add(k, np.tile(v, [1,1]).astype(np.float32))

# Set up model runner
trainer = utils.Trainer(args, add_timestamp=True)
self = Runner(args, 0, trainer) # instantiate an object of Runner
self.params_dict = para_settings
self._prepare_data(data_settings)
self.n_batch = min(self.params_dict['n_batch'], self.dataset_pair.n_train)

# Set various attributes of the model
model = self.params_dict["model"]
model.init_with_params(self.params_dict, self.procdata)

# Define simulation variables and run simulator
times = np.linspace(0.0, 20.0, 101).astype(np.float32) #0-20s,101个time step
conditions = np.array([[1.0, 1.0]]).astype(np.float32)
dev_1hot = np.expand_dims(np.zeros(7).astype(np.float32),0) #multi-hot vector
print(theta)
sol_rk4 = model.simulate(theta, times, conditions, dev_1hot, 'rk4')[0] #Runge-Kutta method
sol_mod = model.simulate(theta, times, conditions, dev_1hot, 'modeulerwhile')[0]

# Run TF session to extract an ODE simulation using modified Euler and RK4
sess = tf.Session()
#sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) 
[mod, rk4] = sess.run([sol_mod, sol_rk4])
print(np.shape(mod))

# Ensure that the relative error is no bigger than 5%
Y0 = mod[0][0][-1]
#print(mod)
print(np.shape(Y0))
Y1 = rk4[0][0][-1]
#print(Y1)
assert np.nanmax(np.abs((Y0 - Y1) / Y0)) < 0.05, 'Difference between Modified Euler and RK4 solvers greater than 5%'
