# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from __future__ import absolute_import
import argparse
import os
import time
from typing import Any, Dict, List

# Special imports for matplotlib because of "use" call
 # pylint: disable=wrong-import-order,wrong-import-position
import matplotlib
matplotlib.use('agg')
# Import of pyplot must be postponed until after call of matplotlib.use.
import matplotlib.pyplot as pp

# Standard data science imports
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Local imports
from src import procdata
from src.parameters import Parameters
from src.plotting import plot_prediction_summary, xval_treatments, plot_weighted_theta, species_summary
from src.convenience import Decoder, Encoder, SessionVariables, LocalAndGlobal, Objective, Placeholders, TrainingLogData, TrainingStepper
from src.xval import XvalMerge
from src import utils

class Runner:
    """A class to set up, train and evaluate a variation CRN model, holding out one fold 
    of a data set for validation. See "run_on_split" below for how to set it up and run it."""

    def __init__(self, args, split=None, trainer=None):
        """
        :param args: a Namespace, from argparse.parse_args
        :param split: an integer between 1 and args.folds inclusive, or None
        :param trainer: a Trainer instance, or None
        """
        self.procdata = None
        # Command-line arguments (Namespace)
        self.args = self._tidy_args(args, split)
        # TODO(dacart): introduce a switch to allow non-GPU use, achieved with:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Utility methods for training a model
        self.trainer = trainer or utils.Trainer(args, add_timestamp=True)
        # Attributes set in other methods:
        # Conditions, input signals of the data that are being modelled
        self.conditions = None
        # DatasetPair, from a training Dataset and validation Dataset
        self.dataset_pair = None
        # Decoder and encoder networks
        self.decoder = None
        self.encoder = None
        # Number of instances in a batch (int)
        self.n_batch = None
        # Number of "theta" parameters: local, global-conditional and global (int)
        self.n_theta = None
        # Collection of attributes related to training objective
        self.objective = None
        # Value of spec["params"] from YAML file (dict)
        self.params_dict = None
        # Collection of placeholder attributes, each a Tensor, fed with new values for each batch
        self.placeholders = None
        # Training feed_dict: dict from placeholder Tensor to np.array
        self.train_feed_dict = None
        # TrainingStepper object
        self.training_stepper = None
        # Validation feed_dict keys: dict from placeholder Tensor to np.array
        self.val_feed_dict = None
        # Model path for storing best weights so far
        self.model_path = os.path.join(self.trainer.tb_log_dir, 'saver', 'sess_max_elbo')

    @classmethod
    def _tidy_args(cls, args, split):
        print("Called run_xval with ")
        print(args)
        if getattr(args, 'heldout', None):
            print("Heldout device is %s" % args.heldout)
        else:
            args.heldout = None # in case not defined at all
            if split is not None:
                args.split = split
            print("split = %d" % args.split)
        if args.epochs < args.test_epoch:
            msg = "No test epochs possible with epochs = %d and test_epochs = %d"%(args.epochs,args.test_epoch)
            raise Exception(msg)
        return args

    def _fix_random_seed(self):
        """Try to fix the TensorFlow and Numpy random seeds to achieve deterministic behaviour.
        Currently (2019-01-14) it doesn't do that, and adding a call to random.seed(seed) doesn't help either."""
        seed = self.args.seed #当在代码中使用了随机数，但是希望代码在不同时间或者不同的机器上运行能够得到相同的随机数，以至于能够得到相同的结果
        print("Setting: tf.set_random_seed({})".format(seed))
        tf.set_random_seed(seed)
        print("Setting: np.random.seed({})".format(seed))
        np.random.seed(seed)

    @staticmethod
    def _decide_indices(size, randomize):
        if randomize:
            return np.random.permutation(size)
        return np.arange(size, dtype=int)

    def _init_chunks(self):
        size = self.dataset_pair.n_train #234
        chunk_size = self.n_batch #36
        indices = self._decide_indices(size, randomize=True) #随机打乱返回一个打乱的，长度为234的np.ndarray
        return [indices[i:i + chunk_size] for i in range(0, size, chunk_size)] #从0数到234，步长为36，返回一个列表，该列表中用7个np.ndarray

    @classmethod
    def _init_fold_chunks(cls, size, folds, randomize):
        indices = cls._decide_indices(size, randomize)
        return np.array_split(indices, folds)

    @staticmethod
    def _decide_dataset_pair(all_ids, val_ids, loaded_data, data_settings):
        train_ids = np.setdiff1d(all_ids, val_ids) #在all_ids，但不在val_ids的已排序的唯一值
        train_data, val_data = procdata.split_by_train_val_ids(loaded_data, train_ids, val_ids)
        return procdata.DatasetPair(procdata.Dataset(data_settings, train_data, train_ids), procdata.Dataset(data_settings, val_data, val_ids))

    def _prepare_data(self, data_settings: Dict[str, str]):
        '''data: a dictionary of the form {'devices': [...], 'files': [...]}
              where the devices are names like Pcat_Y81C76 and the files are csv basenames.
        Sets self.dataset_pair to hold the training and evaluation datasets.'''
        # "Make a usable dataset (notions of repeats and conditions)"
        # Establish where the csv files in data['files'] are to be found.
        data_dir = utils.get_data_directory() #返回存放data的文件夹名称（csv格式）
        # Map data arguments into internal structures
        #self.conditions = data_settings["conditions"]
        # Load all the data in all the files, merging appropriately (see procdata for details).
        self.procdata = procdata.ProcData(data_settings)
        loaded_data = self.procdata.load_all(data_dir) #312 wells, 312 conditions
        np.random.seed(self.args.seed)
        if self.args.heldout: #None (Not None = True)
            # We specified a holdout device to act as the validation set.
            d_train, d_val, train_ids, val_ids = procdata.split_holdout_device(self.procdata, loaded_data, self.args.heldout)
            train_dataset = procdata.Dataset(data_settings, d_train, train_ids)
            val_dataset = procdata.Dataset(data_settings, d_val, val_ids)
            self.dataset_pair = procdata.DatasetPair(train_dataset, val_dataset)
        else:
            # Number of conditions (wells) in the data.
            loaded_data_length = loaded_data['X'].shape[0] # len(loaded_data['X']) #312个数据 （312 differently conditioned samples）
            # The validation set is determined by the values of "split" and args.folds.
            # A list of self.args.folds (roughly) equal size numpy arrays of int, total size W, values in [0, W-1]
            # where W is the well (condition) count, all distinct.
            val_chunks = self._init_fold_chunks(loaded_data_length, self.args.folds, randomize=True)
            assert len(val_chunks) == self.args.folds, "Bad chunks"
            # All the ids from 0 to W-1 inclusive, in order.
            all_ids = np.arange(loaded_data_length, dtype=int) # 0 - 311 总共312个数字
            # split runs from 1 to args.folds, so the index we need is one less.
            # val_ids is the indices of data items to be used as validation data.
            val_ids = np.sort(val_chunks[self.args.split - 1])
            # A DatasetPair object: two Datasets (one train, one val) plus associated information.
            self.dataset_pair = self._decide_dataset_pair(all_ids, val_ids, loaded_data, data_settings)

    def _random_noise(self, size):
        return np.random.randn(size, self.args.test_samples, self.n_theta)

    def _create_feed_dicts(self):
        ph = self.placeholders
        self.train_feed_dict = self.dataset_pair.train.create_feed_dict(ph, self._random_noise(self.dataset_pair.n_train))
        self.val_feed_dict = self.dataset_pair.val.create_feed_dict(ph, self._random_noise(self.dataset_pair.n_val)) #返回一个以placeholder属性维key,具体值维value的字典

    def set_up(self, data_settings, params):    
        self.params_dict = params #object Runner的method: set_up

        # time some things, like epoch time
        start_time = time.time()

        # ---------------------------------------- #
        #     DEFINE XVAL DATASETS                 #
        # ---------------------------------------- #

        # Create self.dataset_pair: DatasetPair containing train and val Datasets.
        self._prepare_data(data_settings)
        # Number of instances to put in a training batch.
        self.n_batch = min(self.params_dict['n_batch'], self.dataset_pair.n_train)#如果训练集样本个数小于一个batch中的样本个数的话 就整个训练样本一起训练了

        # This is already a model object because of the use of "!!python/object:... in the yaml file.
        model = self.params_dict["model"]
        # Set various attributes of the model
        model.init_with_params(self.params_dict, self.procdata)
        
        # Import priors from YAML
        parameters = Parameters() #instantiate a class
        parameters.load(self.params_dict)

        print("----------------------------------------------")
        if self.args.verbose:
            print("parameters:")
            parameters.pretty_print()
        n_vals = LocalAndGlobal.from_list(parameters.get_parameter_counts()) #(10,0,0,4), 有4个属性，分别记录着constant,global,global_conditioned,local parameters的维度
        self.n_theta = n_vals.sum() #所有parameters的个数 总共14个parameters

        #     TENSORFLOW PARTS        #
        self.placeholders = Placeholders(self.dataset_pair, n_vals)

        # feed_dicts are used to supply placeholders, these are for the entire train/val dataset, there is a batch one below.
        self._create_feed_dicts() #是一个method,用于改变对象的属性

        # time-series of species differences: x_delta_obs is BATCH x (nTimes-1) x nSpecies
        x_delta_obs = self.placeholders.x_obs[:, 1:, :] - self.placeholders.x_obs[:, :-1, :] #时间维度做差（第2个到第86个-第一个到第85个）

        # DEFINE THE ENCODER NN: for LOCAL PARAMETERS
        print("Set up encoder")
        self.encoder = Encoder(self.args.verbose, parameters, self.placeholders, x_delta_obs) #self.args.verbose==True

        # DEFINE THE DECODER NN
        print("Set up decoder")
        self.decoder = Decoder(self.params_dict, self.placeholders, self.dataset_pair.times, self.encoder)

        # DEFINE THE OBJECTIVE and GRADIENTS
        # likelihood p (x | theta)
        print("Set up objective")
        self.objective = Objective(self.encoder, self.decoder, model, self.placeholders)

        # SET-UP tensorflow LEARNING/OPTIMIZER
        self.training_stepper = TrainingStepper(self.args.dreg, self.encoder, self.objective, self.params_dict) #类的属性可以使用了 dreg: doubly reparametrized gradient estimator
        time_interval = time.time() - start_time
        print("Time before sess: %g" % time_interval)

        # TENSORBOARD VISUALIZATION            #
        ts_to_vis = 1
        plot_histograms = self.params_dict["plot_histograms"]
        self.encoder.q.attach_summaries(plot_histograms)  # global and local parameters of q distribution
        unnormed_iw = self.objective.log_unnormalized_iws[ts_to_vis, :]
        self_normed_iw = self.objective.normalized_iws[ts_to_vis, :]   # not in log space
        utils.variable_summaries(unnormed_iw, 'IWS_unn_log', plot_histograms)
        utils.variable_summaries(self_normed_iw, 'IWS_normed', plot_histograms)
        tf.summary.scalar('IWS_normed/nonzeros', tf.count_nonzero(self_normed_iw))

        with tf.name_scope('ELBO'):
            tf.summary.scalar('elbo', self.objective.elbo)
            # log(P) and also a per-species breakdown
            log_p = tf.reduce_mean(tf.reduce_logsumexp(self.objective.log_p_observations, axis=1))
            tf.summary.scalar('log_p', log_p)  # [batch, 1]
            for i,plot in enumerate(self.procdata.signals):
                log_p_by_species = tf.reduce_mean(tf.reduce_logsumexp(self.objective.log_p_observations_by_species[:,:,i], axis=1))
                tf.summary.scalar('log_p_'+plot, log_p_by_species)
            # Priors
            logsumexp_log_p_theta = tf.reduce_logsumexp(self.encoder.log_p_theta, axis=1)
            tf.summary.scalar('log_prior', tf.reduce_mean(logsumexp_log_p_theta))
            logsumexp_log_q_theta = tf.reduce_logsumexp(self.encoder.log_q_theta, axis=1)
            tf.summary.scalar('loq_q', tf.reduce_mean(logsumexp_log_q_theta))

    def _create_session_variables(self):
        return SessionVariables([
            self.objective.log_normalized_iws, #Yes
            self.objective.normalized_iws, #Yes
            self.objective.normalized_iws_reshape,#Yes
            self.decoder.x_post_sample, #Yes
            self.decoder.x_sample, #Yes
            self.objective.elbo, #Yes
            self.objective.precisions, #Yes
            self.encoder.theta.get_tensors(), #Yes
            self.encoder.q.get_tensors()]) #Yes

    def _plot_prediction_summary_figure(self, dataset, output, epoch, writer):
        # dataset: self.dataset_pair.train
        # output: training_output
        fig = plot_prediction_summary(self.procdata, self.decoder.names, self.dataset_pair.times, dataset.X,
                                               output.x_post_sample, output.precisions, dataset.devices, output.log_normalized_iws,
                                               '-') #返回一个6*4的子图阵列
        plot_op = utils.make_summary_image_op(fig, 'Summary', 'Summary')
        writer.add_summary(tf.Summary(value=[plot_op]), epoch)
        pp.close(fig)

    def _plot_species_figure(self, dataset, output, epoch, writer):
        iws = np.exp(np.squeeze(output.log_normalized_iws))
        devices = list(range(len(self.procdata.device_names)))
        fig = species_summary(self.procdata, self.decoder.names, dataset.treatments, dataset.devices, self.dataset_pair.times, 
            output.x_sample, iws, devices)
        plot_op = utils.make_summary_image_op(fig, 'Species', 'Species')
        writer.add_summary(tf.Summary(value=[plot_op]), epoch)
        pp.close(fig)

    def _plot_weighted_theta_figure(self, training_output, validation_output, valid_writer, epoch, sample):
        name = 'Theta-Resample' if sample else 'Theta-Uniform'
        theta_fig = plot_weighted_theta(self.procdata, self.encoder.theta_names, training_output.normalized_iws,
                                        training_output.theta_tensors, self.dataset_pair.train.devices,
                                        validation_output.normalized_iws, validation_output.theta_tensors, self.dataset_pair.val.devices,
                                        columns2use=self.params_dict['theta_columns'], sample=sample)
        theta_plot_op = utils.make_summary_image_op(theta_fig, name, 'Theta')
        valid_writer.add_summary(tf.Summary(value=[theta_plot_op]), epoch)
        pp.close(theta_fig)
        pp.close('all')

    def _evaluate_elbo_and_plot(self, beta_val, epoch, eval_tensors, log_data, merged, sess, train_writer, valid_writer, saver):
        print("epoch %4d"%epoch, end='', flush=True)
        log_data.n_test += 1
        test_start = time.time()
        #plot = np.mod(epoch, self.args.plot_epoch) == 0 #每20个epoch打印一次，用于interaction
        plot = np.mod(epoch, 4) == 0
        # Training
        self.train_feed_dict[self.placeholders.u] = np.random.randn(
            self.dataset_pair.n_train, self.args.train_samples, self.n_theta) #(36,200,14)
        training_output = SessionVariables(sess.run(eval_tensors + [merged], feed_dict=self.train_feed_dict))#只需要把值喂入图中,我就能返回所有感兴趣的一个list of tensors
        train_writer.add_summary(training_output.summaries, epoch) #training_output是一个SessionVariables的对象，有和eval_tensors一样的属性
        if plot:
            self._plot_prediction_summary_figure(self.dataset_pair.train, training_output, epoch, train_writer)
            self._plot_species_figure(self.dataset_pair.train, training_output, epoch, train_writer)
        print(" | train (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(training_output.elbo, log_data.total_train_time / epoch, log_data.total_train_time), end='', flush=True)
        train_writer.flush()
        
        # Validation
        self.val_feed_dict[self.placeholders.u] = np.random.randn( #(36,1000,14)
            self.dataset_pair.n_val, self.args.test_samples, self.n_theta)
        validation_output = SessionVariables(sess.run(eval_tensors + [merged], feed_dict=self.val_feed_dict))
        valid_writer.add_summary(validation_output.summaries, epoch)
        if plot:
            self._plot_prediction_summary_figure(self.dataset_pair.val, validation_output, epoch, valid_writer)
            self._plot_species_figure(self.dataset_pair.val, validation_output, epoch, valid_writer)
        log_data.total_test_time += time.time() - test_start
        print(" | val (iwae-elbo = %0.4f, time = %0.2f, total = %0.2f)"%(validation_output.elbo, log_data.total_test_time / log_data.n_test, log_data.total_test_time))
        valid_writer.flush()
        
        if validation_output.elbo > log_data.max_val_elbo:#每20个epoch就要validation一下，如果validation时候elbo的值比现有的最大的elbo还要大，那么就把它设置为max_val_elbo
            log_data.max_val_elbo = validation_output.elbo
            saver.save(sess, self.model_path)
        
        log_data.training_elbo_list.append(training_output.elbo)
        log_data.validation_elbo_list.append(validation_output.elbo)
    
    def _run_batch(self, beta_val, epoch_start, i_batch, log_data):
        # random indices of training data for this batch
        # i_batch = np.random.permutation(n_train)[:n_batch]
        # placeholders.u: standard normals driving everything!
        u_value = np.random.randn(len(i_batch), self.args.train_samples, self.n_theta) #返回一个（36,200,14） (batch_size,n_iwae,14)
        batch_feed_dict = self.dataset_pair.train.create_feed_dict_for_index( #把csv中具体的数值传进来
            self.placeholders, i_batch, beta_val, u_value)
        beta_val = np.minimum(1.0, beta_val * 1.01)
        # keep track of this time, sometimes (not here) this can be inefficient
        log_data.batch_feed_time += time.time() - epoch_start #增加delta_t 给placeholder喂参数需要多长时间
        train_start = time.time()
        # take a training step
        self.training_stepper.train_step.run(feed_dict=batch_feed_dict) #run是否为tensorflow的内置函数？ self.training_stepper.train_step: AdamOptimizer 这就是真正的训练了
        # see how long it took
        log_data.batch_train_time += time.time() - train_start #增加delta_t 根据一个batch(36个samples)更新一次梯度需要多少时间
        return beta_val

    def _run_session(self):
        # tf.summary.scalar('ESS',  1.0 / np.sum(np.square(ws), 1))
        merged = tf.summary.merge_all() #将所有的summary保存到磁盘，以便tensorboard显示
        held_out_name = self.args.heldout or '%d_of_%d' % (self.args.split, self.args.folds) #1_of_4
        train_writer = tf.summary.FileWriter(os.path.join(self.trainer.tb_log_dir, 'train_%s' % held_out_name)) #./unnamed_20201005T181942164282/train_1_of_4
        valid_writer = tf.summary.FileWriter(os.path.join(self.trainer.tb_log_dir, 'valid_%s' % held_out_name)) #./unnamed_20201005T181942164282/valid_1_of_4
        eval_tensors = self._create_session_variables().as_list()
        saver = tf.train.Saver()
        print("----------------------------------------------")
        print("Starting Session...")
        beta = 1.0
        with tf.Session() as sess:
            self._fix_random_seed()  # <-- force run to be deterministic given random seed (所以我们需要set seed for tf and numpy)
            # initialize variables in the graph
            sess.run(tf.global_variables_initializer()) #返回一个用来初始化 计算图中 所有global variable的运算操作
            log_data = TrainingLogData()
            print("===========================")
            if self.args.heldout:
                split_name = 'heldout device = %s' % self.args.heldout
            else:
                split_name = 'split %d of %d' % (self.args.split, self.args.folds)
            print("Training: %s"%split_name) #Training: split 1 of 4
            for epoch in range(1, self.args.epochs + 1): #总共有1000个epochs range(1,1001)也就是从1到1000
                epoch_start = time.time()
                epoch_batch_chunks = self._init_chunks()
                for i_batch in epoch_batch_chunks: #总共7个batch,前6个batch有36个samples,第七个batch有18个samples
                   beta = self._run_batch(beta, epoch_start, i_batch, log_data)
                log_data.total_train_time += time.time() - epoch_start
                # occasionally evaluation ELBO on train and val, using more IW samples
                if np.mod(epoch, 4) == 0:
                #if np.mod(epoch, self.args.test_epoch) == 0: #每20个epoch就打印一次：总共1000个epoch,所以打印50次
                    self._evaluate_elbo_and_plot(beta, epoch, eval_tensors, log_data, merged, sess, train_writer, #tensor for evaluation
                        valid_writer, saver)
            train_writer.close()
            valid_writer.close()
            print("===========================")

            # Apply weights
            # TODO(dacart): do we need the next line?
            #_training_output = SessionVariables(sess.run(eval_tensors, feed_dict=self.train_feed_dict))
            saver.restore(sess, self.model_path)
            validation_output = SessionVariables(sess.run(eval_tensors, feed_dict=self.val_feed_dict))
        sess.close() #当进程结束释放资源（用于Variable和Queue的资源）
        tf.reset_default_graph()
        return log_data, validation_output

    # def _do_merge(self, val_results):
    #     xval_merge = XvalMerge(self.args, self.trainer)
    #     # change to split so names are recoreded
    #     xval_merge.add(1, self.dataset_pair, val_results)
    #     xval_merge.finalize()
    #     xval_merge.make_writer(xval_merge.trainer.tb_log_dir)
    #     xval_merge.make_images(self.procdata)
    #     xval_merge.close_writer()
    #     xval_merge.save(xval_merge.trainer.tb_log_dir)

    def run(self):
        log_data, validation_output = self._run_session() #所有self对象的属性都会通过这个_run_session得到改变
        val_results = {"names": self.decoder.names,
                       "times": self.dataset_pair.times,
                       "x_sample": validation_output.x_sample,
                       "x_post_sample": validation_output.x_post_sample,
                       "precisions": validation_output.precisions,
                       "log_normalized_iws": validation_output.log_normalized_iws,
                       "elbo": validation_output.elbo,
                       "elbo_list": log_data.validation_elbo_list,
                       "theta": validation_output.theta_tensors,
                       "q_names": self.encoder.q.get_tensor_names(),
                       "q_values": validation_output.q_params}
        #if self.args.heldout:
        #    self._do_merge(val_results)
        return self.dataset_pair, val_results


def create_parser(with_split: bool):
    parser = argparse.ArgumentParser(description='VI-HDS')
    #parser.add_argument('yaml', type=str, help='Name of yaml spec file')
    parser.add_argument('--yaml', type=str,default='../specs/dr_constant_precisions.yaml', help='Name of yaml spec file')
    parser.add_argument('--experiment', type=str, default='unnamed', help='Name for experiment, also location of tensorboard and saved results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--test_epoch', type=int, default=20, help='Frequency of calling test')
    parser.add_argument('--plot_epoch', type=int, default=100, help='Frequency of plotting figures')
    parser.add_argument('--train_samples', type=int, default=200, help='Number of samples from q, per datapoint, during training')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of samples from q, per datapoint, during testing')
    parser.add_argument('--dreg', type=bool, default=True, help='Use DReG estimator') #If true, then use dreg (doubly reparametrized gradient estimator)
    parser.add_argument('--verbose', action='store_true', default=True, help='Print more information about parameter setup')
    if with_split:
        # We make --heldout (heldout device) and --split mutually exclusive. Really we'd like to say it's allowed
        # to specify *either* --heldout *or* --split and/or --folds, but argparse isn't expressive enough for that.
        # So if the user specifies --heldout and --folds, there won't be a complaint here, but --folds will be ignored.
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--heldout', type=str, help='name of held-out device, e.g. R33S32_Y81C76')
        group.add_argument('--split', type=int, default=1, help='Specify split in 1:folds for cross-validation')
    parser.add_argument('--folds', type=int, default=4, help='Cross-validation folds')
    return parser

def run_on_split(args, data_settings, para_settings, split=None, trainer=None):
    runner = Runner(args, split, trainer)
    runner.set_up(data_settings, para_settings)
    return runner.run()

def main():
    parser = create_parser(True)
    print(parser)
    args = parser.parse_args()
    print(args)

    spec = utils.load_config_file(args.yaml)  # spec is a dict of dicts of dicts
    data_settings = procdata.apply_defaults(spec["data"])
    print(data_settings)
    para_settings = utils.apply_defaults(spec["params"])
    xval_merge = XvalMerge(args, data_settings)
    data_pair, val_results = run_on_split(args, data_settings, para_settings, split=None, trainer=xval_merge.trainer)
    xval_merge.add(1, data_pair, val_results)
    xval_merge.finalize()
    xval_merge.save()

    #xval_merge.make_writer(xval_merge.trainer.tb_log_dir)
    #xval_merge.prepare_treatment()
    
    #xval_merge.make_images(data)
    #xval_merge.close_writer()

if __name__ == "__main__": #在其他包导入(import)这个package的时候，main函数不会运行
    main()