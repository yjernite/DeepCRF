import tensorflow as tf
import tensorflow.python.platform
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

from utils import *

###################################
# Register gradients for my ops   #
###################################

@tf.RegisterGradient("ChainCRF")
def _chain_crf_grad(op, grad_likelihood, grad_marginals):
    my_grads = grad_likelihood * op.inputs[4]
    return [my_grads, None, None, None, None]  # List of one Tensor, since we have one input

@tf.RegisterGradient("ChainSumProduct")
def _chain_sum_product_grad(op, grad_forward_sp, grad_backward_sp, grad_gradients):
    return [None, None]

@tf.RegisterGradient("ChainMaxSum")
def _chain_sum_product_grad(op, grad_forward_ms, grad_backward_ms, grad_tagging):
    return [None, None]


###################################
# Auxiliary functions             #
###################################

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name+'_W')


def bias_variable(shape, name='weight'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name+'_b')


###################################
# NN layers                       #
###################################
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


# TODO
def distance_dependent(in_layer, config, params, reuse=False):
    conv_window = config.conv_window
    output_size = config.conv_dim
    batch_size = config.batch_size # int(in_layer.get_shape()[0])
    input_size = config.features_dim #int(in_layer.get_shape()[2])
    moved = [0] * conv_window
    for i in range(conv_window):
        moved[i] = tf.pad(in_layer, [[0, 0], [i, 0], [0, 0]])
        moved[i] = tf.slice(moved[i], [0, 0, 0], [-1, -1, -1])
    


def convo_layer(in_layer, config, params, reuse=False, name='Convo'):
    conv_window = config.conv_window
    output_size = config.conv_dim
    batch_size = config.batch_size # int(in_layer.get_shape()[0])
    input_size = config.features_dim #int(in_layer.get_shape()[2])
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
    reshaped = tf.reshape(in_layer, [batch_size, -1, 1, input_size])
    conv_layer = tf.reshape(conv2d(reshaped, W_conv),
                            [batch_size, -1, output_size], name=name) + b_conv
    return (conv_layer, W_conv, b_conv)


###################################
# Potentials layers               #
###################################


# takes features and outputs potentials
def potentials_layer(in_layer, mask, config, params, reuse=False, name='Potentials'):
    num_steps = int(in_layer.get_shape()[1])
    input_size = int(in_layer.get_shape()[2])
    pot_shape = [config.n_tags] * config.pot_size
    out_shape = [config.batch_size, config.num_steps] + pot_shape
    pot_card = config.n_tags ** config.pot_size
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_pot = params.W_pot
        b_pot = params.b_pot
    else:
        W_pot = weight_variable([input_size, pot_card], name=name)
        b_pot = bias_variable([pot_card], name=name)
        W_pot = tf.clip_by_norm(W_pot, config.param_clip)
        b_pot = tf.clip_by_norm(b_pot, config.param_clip)
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.matmul(flat_input, W_pot) + b_pot
    pots_layer = tf.reshape(pre_scores, out_shape)
    # define potentials for padding tokens
    padding_pot = np.zeros(pot_shape)
    padding_pot[..., 0] += 1e2
    padding_pot -= 1e2
    pad_pot = tf.convert_to_tensor(padding_pot, tf.float32)
    pad_pots = tf.expand_dims(tf.expand_dims(pad_pot, 0), 0)
    pad_pots = tf.tile(pad_pots,
                       [config.batch_size,
                        config.num_steps] + [1] * config.pot_size)
    # expand mask
    mask_a = mask
    for _ in range(config.pot_size):
        mask_a = tf.expand_dims(mask_a, -1)
    mask_a = tf.tile(mask_a, [1, 1] + pot_shape)
    # combine
    pots_layer = (pots_layer * mask_a + (1 - mask_a) * pad_pots)
    return (pots_layer, W_pot, b_pot)


# alternatively: unary + binary
def binary_log_pots(in_layer, config, params, reuse=False, name='Binary'):
    input_size = int(in_layer.get_shape()[2])
    pot_shape = [config.n_tags] * 2
    out_shape = [config.batch_size, -1] + pot_shape
    pot_card = config.n_tags ** 2
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_pot_bin = params.W_pot_bin
        b_pot_bin = params.b_pot_bin
    else:
        W_pot_bin = weight_variable([input_size, pot_card], name=name)
        b_pot_bin = bias_variable([pot_card], name=name)
        W_pot_bin = tf.clip_by_norm(W_pot_bin, config.param_clip)
        b_pot_bin = tf.clip_by_norm(b_pot_bin, config.param_clip)
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.matmul(flat_input, W_pot_bin) + b_pot_bin
    bin_pots_layer = tf.reshape(pre_scores, out_shape)
    return (bin_pots_layer, W_pot_bin, b_pot_bin)


def unary_log_pots(in_layer, mask, config, params, reuse=False, name='Unary'):
    input_size = int(in_layer.get_shape()[2])
    pot_shape = [config.n_tags]
    out_shape = [config.batch_size, -1] + pot_shape
    pot_card = config.n_tags
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_pot_un = params.W_pot_un
        b_pot_un = params.b_pot_un
    else:
        W_pot_un = weight_variable([input_size, pot_card], name=name)
        b_pot_un = bias_variable([pot_card], name=name)
        W_pot_un = tf.clip_by_norm(W_pot_un, config.param_clip)
        b_pot_un = tf.clip_by_norm(b_pot_un, config.param_clip)
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.matmul(flat_input, W_pot_un) + b_pot_un
    un_pots_layer = tf.reshape(pre_scores, out_shape)
    # define potentials for padding tokens
    shape_aux = 0 * mask + 1
    pad_pots = tf.pack([0 * shape_aux] + [-1e2 * shape_aux
                                          for _ in range(config.n_tags - 1)])
    pad_pots = tf.transpose(pad_pots, [1, 2, 0])
    # expand mask
    mask_a = tf.expand_dims(mask, -1)
    mask_a = tf.tile(mask_a, [1, 1] + pot_shape)
    # combine
    un_pots_layer = (un_pots_layer * mask_a + (1 - mask_a) * pad_pots)
    return (un_pots_layer, W_pot_un, b_pot_un)


def log_pots(un_pots_layer, bin_pots_layer, config, params, name='LogPotentials'):
    expanded_unaries = tf.expand_dims(un_pots_layer, 2)
    expanded_unaries = tf.tile(expanded_unaries, [1, 1, config.n_tags, 1])
    pots_layer = expanded_unaries + bin_pots_layer
    return pots_layer


###################################
# Objective layers                #
###################################


# Takes a representation as input and returns predictions of tag windows
def predict_layer(in_layer, config, params, reuse=False, name='Predict'):
    n_outcomes = config.n_outcomes
    batch_size = config.batch_size
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
    preds_layer = tf.reshape(pre_scores, [batch_size, -1, config.n_tag_windows])
    return (preds_layer, W_pred, b_pred)


# Takes tag window predictions, and returns cross-entropy and accuracy
def optim_outputs(outcome, targets, config, params):
    batch_size = int(outcome.get_shape()[0])
    n_outputs = int(outcome.get_shape()[2])
    # We are currently using cross entropy as criterion
    cross_entropy = tf.reduce_sum(targets * tf.log(outcome))
    # We also compute the per-tag accuracy
    correct_prediction = tf.equal(tf.argmax(outcome, 2), tf.argmax(targets, 2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                     "float") * tf.reduce_sum(targets, 2)) /\
        tf.reduce_sum(targets)
    return (cross_entropy, accuracy)


# Takes potentials and markov blanket indices (pot_indices) and returns
# the pseudo-likelihood criterion
def pseudo_likelihood(potentials, pot_indices, targets, config):
    pots_shape = map(int, potentials.get_shape()[2:])
    # make pots
    reshaped = [None] * config.pot_size
    for i in range(config.pot_size):
        reshaped[i] = potentials
        multiples = [1] * (2 * config.pot_size + 1)
        for j in range(i):
            reshaped[i] =  tf.expand_dims(reshaped[i], 2)
            multiples[2 + j] = config.n_tags
        for j in range(config.pot_size - i - 1):
            reshaped[i] =  tf.expand_dims(reshaped[i], -1)
            multiples[-1 - j] = config.n_tags
        reshaped[i] = tf.tile(reshaped[i], multiples[:])
        paddings = [[0, 0], [i, config.pot_size - i - 1]] + [[0, 0]] * (2 * config.pot_size - 1)
        reshaped[i] = tf.reshape(tf.pad(reshaped[i], paddings),
                                 [config.batch_size,
                                  config.num_steps + config.pot_size - 1,
                                  -1])
    pre_cond = tf.reduce_sum(tf.pack(reshaped), 0)
    # print pre_cond.get_shape()
    begin_slice = [0, 0, 0]
    end_slice = [-1, config.num_steps, -1]
    pre_cond = tf.slice(pre_cond, begin_slice, end_slice)
    pre_cond = tf.reshape(pre_cond, [config.batch_size, config.num_steps] +
                                          [config.n_tags] * (2 * config.pot_size - 1))
    # print pre_cond.get_shape()
    # move the current tag to the last dimension
    perm = range(len(pre_cond.get_shape()))
    perm[-1] = perm[-config.pot_size]
    for i in range(0, config.pot_size -1):
        perm[-config.pot_size + i] = perm[-config.pot_size + i] + 1
    perm_potentials = tf.transpose(pre_cond, perm=perm)
    # get conditional distribution of the current tag
    flat_pots = tf.reshape(perm_potentials, [-1, config.n_tags])
    flat_cond = tf.gather(flat_pots, pot_indices)
    pre_shaped_cond = tf.nn.softmax(flat_cond)
    conditional = tf.reshape(pre_shaped_cond, [config.batch_size, config.num_steps, -1])
    # compute pseudo-log-likelihood of sequence
    p_ll = tf.reduce_sum(targets * tf.log(conditional + 1e-25)) # avoid underflow
    return (conditional, p_ll)


# Takes potentials an returns log-likelihood
def log_score(potentials, window_indices, mask, config):
    batch_size = int(potentials.get_shape()[0])
    num_steps = int(potentials.get_shape()[1])
    pots_shape = map(int, potentials.get_shape()[2:])
    flat_pots = tf.reshape(potentials, [-1])
    flat_scores = tf.gather(flat_pots,
                            window_indices / (config.n_tags ** (config.pot_size - 1)))
    scores = tf.reshape(flat_scores, [batch_size, num_steps])
    scores = tf.mul(scores, mask)
    return tf.reduce_sum(scores)


###################################
# Making a (deep) CRF             #
###################################
class CRF:
    def __init__(self, config):
        self.batch_size = config.batch_size
        num_features = len(config.input_features)
        # input_ids <- batch.features
        self.input_ids = tf.placeholder(tf.int32)
        # mask <- batch.mask
        self.mask = tf.placeholder(tf.float32)
        # pot_indices <- batch.tag_neighbours_lin
        self.pot_indices = tf.placeholder(tf.int32)
        # tags <- batch.tags
        self.tags = tf.placeholder(tf.int32)
        # targets <- batch.tags_one_hot
        self.targets = tf.placeholder(tf.float32)
        # window_indices <- batch.tag_windows_lin
        self.window_indices = tf.placeholder(tf.int32)
        # nn_targets <- batch.tag_windows_one_hot
        self.nn_targets = tf.placeholder(tf.float32)
    
    def make(self, config, params, reuse=False, name='CRF'):
        with tf.variable_scope(name):
            self.l1_norm = tf.reduce_sum(tf.zeros([1]))
            self.l2_norm = tf.reduce_sum(tf.zeros([1]))
            ### EMBEDDING LAYER
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # initial embedding
            (out_layer, embeddings) = feature_layer(self.input_ids,
                                                    config, params,
                                                    reuse=reuse)
            params.embeddings = embeddings
            for feat in config.l1_list:
                self.l1_norm += L1_norm(params.embeddings[feat])
            for feat in config.l2_list:
                self.l2_norm += L2_norm(params.embeddings[feat])
            if config.verbose:
                print('features layer done')
            # convolution
            if config.use_convo:
                if config.bi_convo:
                    (out_layer, W_pre_conv, b_pre_conv) = convo_layer(out_layer,
                                                                      config,
                                                                      params)
                    self.l2_norm += L2_norm(W_pre_conv) + L2_norm(b_pre_conv)
                    out_layer = tf.nn.relu(out_layer)
                (out_layer, W_conv, b_conv) = convo_layer(out_layer, config,
                                                          params, reuse=reuse)
                params.W_conv = W_conv
                params.b_conv = b_conv
                # self.l1_norm += L1_norm(W_conv) + L1_norm(b_conv)
                self.l2_norm += L2_norm(W_conv) + L2_norm(b_conv)
                if config.verbose:
                    print('convolution layer done')
                out_layer = tf.nn.tanh(out_layer)
            self.out_layer = out_layer
            ### SEQU-NN
            if config.nn_obj_weight > 0:
                (preds_layer, W_pred, b_pred) = predict_layer(out_layer, config,
                                                              params, reuse=reuse)
                params.W_pred = W_pred
                params.b_pred = b_pred
                # self.l1_norm += L1_norm(W_pred) + L1_norm(b_pred)
                self.l2_norm += L2_norm(W_pred) + L2_norm(b_pred)
                self.preds_layer = preds_layer
                (cross_entropy, accu_nn) = optim_outputs(preds_layer,
                                                          self.nn_targets,
                                                          config, params)
            ### CRF
            # potentials
            (bin_pots, W_p_b, b_p_b) = binary_log_pots(out_layer, config,
                                                       params, reuse=reuse)
            params.W_pot_bin = W_p_b
            params.b_pot_bin = b_p_b
            self.l2_norm += L2_norm(W_p_b) + L2_norm(b_p_b)
            (un_pots, W_p_u, b_p_u) = unary_log_pots(out_layer, self.mask, config,
                                                     params, reuse=reuse)
            params.W_pot_un = W_p_u
            params.b_pot_un = b_p_u
            self.l2_norm += L2_norm(W_p_u) + L2_norm(b_p_u)
            pots_layer = log_pots(un_pots, bin_pots, config, params)
            if config.verbose:
                print('potentials layer done')
            self.pots_layer = pots_layer
            # log-likelihood, tensor to list
            pots_list = tf.split(0, config.batch_size, self.pots_layer)
            pots_list = [tf.squeeze(pots) for pots in pots_list]
            tags_list = tf.split(0, config.batch_size, self.tags)
            tags_list = [tf.squeeze(tags) for tags in tags_list]
            args_list = zip(pots_list, tags_list)
            # log-likelihood, dynamic programming
            dynamic = [tf.user_ops.chain_sum_product(pots, tags)
                       for pots, tags in args_list]
            pre_crf_list = [(pots, tags, f_sp, b_sp, grads)
                            for ((pots, tags), (f_sp, b_sp, grads)) in zip(args_list, dynamic)]
            crf_list = [tf.user_ops.chain_crf(pots, tags, f_sp, b_sp, grads)
                        for (pots, tags, f_sp, b_sp, grads) in pre_crf_list]
            # log-likelihood, compute
            log_likelihoods = tf.pack([ll for (ll, marg) in crf_list])
            log_likelihood = tf.reduce_sum(log_likelihoods)
            self.log_likelihood = log_likelihood
            self.marginals = tf.pack([marg for (ll, marg) in crf_list])
            # map assignment and accuracy of map assignment
            map_tagging = [tf.user_ops.chain_max_sum(pots, tags)
                           for pots, tags in args_list]
            map_tagging = tf.pack([tging for f_ms, b_ms, tging in map_tagging])
            correct_pred = tf.equal(tf.argmax(map_tagging, 2),
                                    tf.argmax(self.targets, 2))
            correct_pred = tf.cast(correct_pred, "float")
            accuracy = tf.reduce_sum(correct_pred * tf.reduce_sum(self.targets, 2)) /\
                       tf.reduce_sum(self.targets)
            self.map_tagging = map_tagging
            self.accuracy = accuracy
            ### OPTIMIZATION
            # different criteria
            self.criteria = {}
            self.criteria['likelihood'] = -self.log_likelihood
            norm_penalty = config.l1_reg * self.l1_norm + config.l2_reg * self.l2_norm
            for k in self.criteria:
                self.criteria[k] += norm_penalty
                if config.nn_obj_weight > 0:
                    self.criteria[k] -= (config.nn_obj_weight * cross_entropy)
            # corresponding training steps, gradient clipping
            optimizers = {}
            for k in self.criteria:
                if config.optimizer == 'adagrad':
                    optimizers[k] = tf.train.AdagradOptimizer(config.learning_rate,
                                                              name='adagrad_' + k)
                elif config.optimizer == 'adam':
                    optimizers[k] = tf.train.AdamOptimizer(config.learning_rate,
                                                              name='adam_' + k)
                else:
                    optimizers[k] = tf.train.GradientDescentOptimizer(config.learning_rate,
                                                              name='sgd_' + k)
            grads_and_vars = {}
            # gradient clipping
            for k, crit in self.criteria.items():
                uncapped_g_v = optimizers[k].compute_gradients(crit,
                                                               tf.trainable_variables())
                grads_and_vars[k] = [(tf.clip_by_norm(g, config.gradient_clip), v) if g else (g, v)
                                     for g, v in uncapped_g_v]
            self.train_steps = {}
            for k, g_v in grads_and_vars.items():
                self.train_steps[k] = optimizers[k].apply_gradients(g_v)
    
    def train_epoch(self, data, config, params):
        batch_size = config.batch_size
        criterion = self.criteria[config.criterion]
        train_step = self.train_steps[config.criterion]
        # TODO: gradient clipping
        total_crit = 0.
        n_batches = len(data) / batch_size
        batch = Batch()
        for i in range(n_batches):
            batch.read(data, i * batch_size, config, fill=True)
            f_dict = make_feed_crf(self, batch)
            if config.verbose and (i == 0):
                print('First crit: %f' % (criterion.eval(feed_dict=f_dict),))
            train_step.run(feed_dict=f_dict)
            crit = criterion.eval(feed_dict=f_dict)
            total_crit += crit
            if config.verbose and i % 50 == 0:
                train_accuracy = self.accuracy.eval(feed_dict=f_dict)
                print("step %d of %d, training accuracy %f, criterion %f, ll %f, l1 %f, l2 %f" %
                      (i, n_batches, train_accuracy, crit,
                       self.log_likelihood.eval(feed_dict=f_dict),
                       self.l1_norm.eval(), self.l2_norm.eval()))
        print 'total crit', total_crit / n_batches
        return total_crit / n_batches
    
    def validate_accuracy(self, data, config):
        batch_size = config.batch_size
        batch = Batch()
        total_accuracy = 0.
        total_ll = 0.
        total = 0.
        for i in range(len(data) / batch_size):
            batch.read(data, i * batch_size, config, fill=True)
            f_dict = make_feed_crf(self, batch)
            dev_accuracy = self.accuracy.eval(feed_dict=f_dict)
            ll = self.log_likelihood.eval(feed_dict=f_dict)
            total_accuracy += dev_accuracy
            total_ll += ll
            total += 1
            if i % 100 == 0 and config.verbose:
                print("%d of %d: \t map acc: %f \t ll:  %f" % (i, len(data) / batch_size,
                                                total_accuracy / total, total_ll / total))
        return (total_accuracy / total)

