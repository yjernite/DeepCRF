import numpy as np

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

from bi_rnn import bi_rnn
from utils import *

###############################################
# NN creation functions                       #
###############################################
def device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name+'_W')


def bias_variable(shape, name='weight'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name+'_b')


def feature_layer(in_layer, config, params, reuse=False):
    in_features = config.input_features
    features_dim = config.features_dim
    batch_size = config.batch_size
    feature_mappings = config.feature_maps
    # inputs
    num_features = len(in_features)
    input_ids = in_layer
    if reuse:
        tf.get_variable_scope().reuse_variables()
        param_vars = params.embeddings
    # lookup layer
    else:
        param_dic = params.init_dic
        param_vars = {}
        for feat in in_features:
            if feat in param_dic:
                embeddings = \
                      tf.Variable(tf.convert_to_tensor(param_dic[feat],
                                                       dtype=tf.float32),
                                  name=feat + '_embedding',
                                  trainable=False)
                initial = tf.truncated_normal([int(embeddings.get_shape()[1]),
                                               features_dim], stddev=0.1)
                transform_matrix = tf.Variable(initial,
                                               name=feat + '_transform')
                clipped_transform = tf.clip_by_norm(transform_matrix,
                                                    config.param_clip)
                param_vars[feat] = tf.matmul(embeddings, clipped_transform)
            else:
                shape = [len(feature_mappings[feat]['reverse']), features_dim]
                initial = tf.truncated_normal(shape, stddev=0.1)
                param_vars[feat] = tf.Variable(initial,
                                               name=feat + '_embedding')
    params = [param_vars[feat] for feat in in_features]
    input_embeddings = tf.nn.embedding_lookup(params, input_ids,
                                              name='lookup')
    # add and return
    if config.combine == 'sum':
        embedding_layer = tf.reduce_sum(input_embeddings, 2)
    else:
        embedding_layer = tf.reduce_max(input_embeddings, 2)
    return (embedding_layer, param_vars)


def bi_lstm_layer(in_layer, config, reuse=False, name='Bi_LSTM'):
    num_units = config.rnn_hidden_units
    output_size = config.rnn_output_size
    batch_size = int(in_layer.get_shape()[0])
    num_steps = int(in_layer.get_shape()[1])
    input_size = int(in_layer.get_shape()[2])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    lstm_cell_f = rnn_cell.LSTMCell(num_units, input_size, use_peepholes=True,
                                    num_proj=output_size, cell_clip=1.0,
                                    initializer=initializer)
    lstm_cell_b = rnn_cell.LSTMCell(num_units, input_size, use_peepholes=True,
                                    num_proj=output_size, cell_clip=1.0,
                                    initializer=initializer)
    initial_state_f = lstm_cell_f.zero_state(batch_size, tf.float32)
    inputs_list = [tf.reshape(x, [batch_size, input_size])
                   for x in tf.split(1, num_steps, in_layer)]
    rnn_out, rnn_states = bi_rnn(lstm_cell_f, lstm_cell_b, inputs_list,
                                 initial_state=initial_state_f, scope=name,
                                 reuse=reuse)
    out_layer = tf.transpose(tf.pack(rnn_out), perm=[1, 0, 2])
    return out_layer


def convo_layer(in_layer, config, params, reuse=False, name='Convo'):
    conv_window = config.conv_window
    output_size = config.conv_dim
    batch_size = int(in_layer.get_shape()[0])
    num_steps = int(in_layer.get_shape()[1])
    input_size = int(in_layer.get_shape()[2])
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_conv = params.W_conv
        b_conv = params.b_conv
    else:
        W_conv = weight_variable([conv_window, 1, input_size, output_size],
                                 name=name)
        b_conv = bias_variable([output_size], name=name)
        W_conv = tf.clip_by_norm(W_conv, config.param_clip)
        b_conv = tf.clip_by_norm(b_conv, config.param_clip)
    reshaped = tf.reshape(in_layer, [batch_size, num_steps, 1, input_size])
    conv_layer = tf.nn.relu(tf.reshape(conv2d(reshaped, W_conv),
                                       [batch_size, num_steps, output_size],
                                       name=name) + b_conv)
    return (conv_layer, W_conv, b_conv)


def predict_layer(in_layer, config, params, reuse=False, name='Predict'):
    n_outcomes = config.n_outcomes
    batch_size = int(in_layer.get_shape()[0])
    num_steps = int(in_layer.get_shape()[1])
    input_size = int(in_layer.get_shape()[2])
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_pred = params.W_pred
        b_pred = params.b_pred
    else:
        W_pred = weight_variable([input_size, n_outcomes], name=name)
        b_pred = bias_variable([n_outcomes], name=name)
        W_pred = tf.clip_by_norm(W_pred, config.param_clip)
        b_pred = tf.clip_by_norm(b_pred, config.param_clip)
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.nn.softmax(tf.matmul(flat_input, W_pred) + b_pred)
    preds_layer = tf.reshape(pre_scores, [batch_size, num_steps, -1])
    return (preds_layer, W_pred, b_pred)


def optim_outputs(outcome, targets, config, params):
    batch_size = int(outcome.get_shape()[0])
    num_steps = int(outcome.get_shape()[1])
    n_outputs = int(outcome.get_shape()[2])
    # We are currently using cross entropy as criterion
    cross_entropy = tf.reduce_sum(targets * tf.log(outcome))
    # We also compute the per-tag accuracy
    correct_prediction = tf.equal(tf.argmax(outcome, 2), tf.argmax(targets, 2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                     "float") * tf.reduce_sum(targets, 2)) /\
        tf.reduce_sum(targets)
    return (cross_entropy, accuracy)


class SequNN:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        num_features = len(config.input_features)
        # input_ids <- batch.features
        self.input_ids = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                         self.num_steps,
                                                         num_features])
        # targets <- batch.tag_windows_one_hot
        self.targets = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                         self.num_steps,
                                                         config.n_outcomes])
    
    def make(self, config, params, reuse=False, name='SequNN'):
        with tf.variable_scope(name):
            self.l1_norm = tf.reduce_sum(tf.zeros([1]))
            self.l2_norm = tf.reduce_sum(tf.zeros([1]))
            if reuse:
                tf.get_variable_scope().reuse_variables()
            (out_layer, embeddings) = feature_layer(self.input_ids, config,
                                                    params, reuse=reuse)
            params.embeddings = embeddings
            for feat in config.l1_list:
                self.l1_norm += L1_norm(params.embeddings[feat])
            for feat in config.l2_list:
                self.l2_norm += L2_norm(params.embeddings[feat])
            if config.verbose:
                print('features layer done')
            if config.use_rnn:
                out_layer = bi_lstm_layer(embedding_layer, config, reuse=reuse)
                if config.verbose:
                    print('rnn layer done')
            if config.use_convo:
                (out_layer, W_conv, b_conv) = convo_layer(out_layer, config,
                                                          params, reuse=reuse)
                params.W_conv = W_conv
                params.b_conv = b_conv
                # self.l1_norm += L1_norm(W_conv) + L1_norm(b_conv)
                self.l2_norm += L2_norm(W_conv) + L2_norm(b_conv)
                if config.verbose:
                    print('convolution layer done')
            self.out_layer = out_layer
            (preds_layer, W_pred, b_pred) = predict_layer(out_layer, config,
                                                          params, reuse=reuse)
            params.W_pred = W_pred
            params.b_pred = b_pred
            # self.l1_norm += L1_norm(W_pred) + L1_norm(b_pred)
            self.l2_norm += L2_norm(W_pred) + L2_norm(b_pred)
            self.preds_layer = preds_layer
            (cross_entropy, accuracy) = optim_outputs(preds_layer, self.targets, config, params)
            if config.verbose:
                print('output layer done')
            self.accuracy = accuracy
            norm_penalty = config.l1_reg * self.l1_norm + config.l2_reg * self.l2_norm
            criterion = -cross_entropy + norm_penalty
            self.criterion = criterion
            # optimization
            if config.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(config.learning_rate,
                                                      name='adagrad')
            elif config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(config.learning_rate,
                                                   name='adam')
            uncapped_g_v = optimizer.compute_gradients(self.criterion,
                                                       tf.trainable_variables())
            grads_and_vars = [(tf.clip_by_norm(g, config.gradient_clip), v) if g else (g, v)
                              for g, v in uncapped_g_v]
            self.train_step = optimizer.apply_gradients(grads_and_vars)
    
    def train_epoch(self, data, config, params):
        batch_size = config.batch_size
        batch = Batch()
        for i in range(len(data) / batch_size):
            batch.read(data, i * batch_size, config)
            f_dict = {self.input_ids: batch.features,
                      self.targets: batch.tag_windows_one_hot}
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict=f_dict)
                print("step %d of %d, training accuracy %f, l1 %f, l2 %f" %
                      (i, len(data) / batch_size, train_accuracy,
                       self.l1_norm.eval(), self.l2_norm.eval()))
            self.train_step.run(feed_dict=f_dict)
    
    def validate_accuracy(self, data, config):
        batch_size = config.batch_size
        batch = Batch()
        total_accuracy = 0.
        total = 0.
        for i in range(len(data) / batch_size):
            batch.read(data, i * batch_size, config)
            f_dict = {self.input_ids: batch.features,
                      self.targets: batch.tag_windows_one_hot}
            dev_accuracy = self.accuracy.eval(feed_dict=f_dict)
            total_accuracy += dev_accuracy
            total += 1
            if i % 100 == 0:
                print("%d of %d: \t:%f" % (i, len(data) / batch_size,
                                           total_accuracy / total))
        return total_accuracy / total





