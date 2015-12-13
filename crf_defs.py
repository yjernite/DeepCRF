from nn_defs import *
from utils import *
from tensorflow.models.rnn.rnn_cell import *

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

###################################
# Building blocks                 #
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
        W_pot = tf.clip_by_norm(W_pot, 10)
        b_pot = tf.clip_by_norm(b_pot, 10)
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


# pseudo-likelihood criterion
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
    print pre_cond.get_shape()
    begin_slice = [0, 0, 0]
    end_slice = [-1, config.num_steps, -1]
    pre_cond = tf.slice(pre_cond, begin_slice, end_slice)
    pre_cond = tf.reshape(pre_cond, [config.batch_size, config.num_steps] +
                                          [config.n_tags] * (2 * config.pot_size - 1))
    print pre_cond.get_shape()
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


# dynamic programming part 1: max sum
class CRFMaxCell(RNNCell):
    """Dynamic programming for CRF"""
    def __init__(self, config):
        self._num_units = config.n_tags ** (config.pot_size - 1)
        self.n_tags = config.n_tags
    
    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
    
    @property
    def state_size(self):
        return self._num_units
    
    def __call__(self, inputs, state, scope=None):
        """Summation for dynamic programming. Inputs are the
        log-potentials. States are the results of the summation at the
        last step"""
        with tf.variable_scope(scope or type(self).__name__):
            # add states and log-potentials
            multiples = [1] * (len(state.get_shape()) + 1)
            multiples[-1] = self.n_tags
            exp_state = tf.tile(tf.expand_dims(state, -1), multiples)
            added = exp_state + inputs
            # return maxes, arg_maxes along first dimension (after the batch dim)
            new_state = tf.reduce_max(added, 1)
            max_id = tf.argmax(added, 1)
        return new_state, max_id


# max a posteriori tags assignment: implement dynamic programming
def map_assignment(potentials, config):
    pots_shape = map(int, potentials.get_shape()[2:])
    inputs_list = [tf.reshape(x, [config.batch_size] + pots_shape)
                   for x in tf.split(1, config.num_steps, potentials)]
    # forward pass
    max_cell = CRFMaxCell(config)
    max_ids = [None] * len(inputs_list)
    # initial state: starts at 0 - 0 - 0 etc...
    state = tf.zeros(pots_shape[:-1])
    for t, input_ in enumerate(inputs_list):
        state, max_id = max_cell(inputs_list[t], state)
        max_ids[t] = max_id
    # backward pass
    powers = tf.to_int64(map(float, range(config.batch_size))) * \
             (config.n_tags ** (config.pot_size - 1))
    outputs = [-1] * len(inputs_list)
    best_end = tf.argmax(tf.reshape(state, [config.batch_size, -1]), 1)
    current = best_end
    max_pow = (config.n_tags ** (config.pot_size - 2))
    for i, _ in enumerate(outputs):
        outputs[-1 - i] = (current / max_pow) 
        prev_best = tf.gather(tf.reshape(max_ids[-1 - i], [-1]), current + powers)
        current = prev_best * max_pow + (current / config.n_tags)
    map_tags = tf.transpose(tf.pack(outputs))
    return map_tags


# compute the log to get the log-likelihood
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
        self.num_steps = config.num_steps
        num_features = len(config.input_features)
        # input_ids <- batch.features
        self.input_ids = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                         self.num_steps,
                                                         num_features])
        # mask <- batch.mask
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
        # pot_indices <- batch.tag_neighbours_lin
        self.pot_indices = tf.placeholder(tf.int32,
                                          [config.batch_size * config.num_steps])
        # tags <- batch.tags
        self.tags = tf.placeholder(tf.int32,
                                   [config.batch_size, config.num_steps])
        # targets <- batch.tags_one_hot
        self.targets = tf.placeholder(tf.float32, [config.batch_size,
                                                   config.num_steps,
                                                   config.n_tags])
        # window_indices <- batch.tag_windows_lin
        self.window_indices = tf.placeholder(tf.int32,
                                             [config.batch_size * config.num_steps])

    def make(self, config, params, reuse=False, name='CRF'):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # out_layer <- output of NN (TODO: add layers)
            (out_layer, embeddings) = feature_layer(self.input_ids,
                                                    config, params,
                                                    reuse=reuse)
            params.embeddings = embeddings
            if config.verbose:
                print('features layer done')
            # convolution
            if config.use_convo:
                (out_layer, W_conv, b_conv) = convo_layer(out_layer, config,
                                                          params, reuse=reuse)
                params.W_conv = W_conv
                params.b_conv = b_conv
                #~ (out_layer, W_conv, b_conv) = convo_layer(out_layer, config,
                                                          #~ params, reuse=reuse)
                if config.verbose:
                    print('convolution layer done')
            self.out_layer = out_layer
            # pots_layer <- potentials
            (pots_layer, W_pot, b_pot) = potentials_layer(out_layer,
                                                          self.mask,
                                                          config, params,
                                                          reuse=reuse)
            params.W_pot = W_pot
            params.b_pot = b_pot
            if config.verbose:
                print('potentials layer done')
            self.pots_layer = pots_layer
            # pseudo-log-likelihood
            conditional, pseudo_ll = pseudo_likelihood(pots_layer,
                                                       self.pot_indices,
                                                       self.targets, config)
            self.pseudo_ll = pseudo_ll
            # accuracy of p(t_i | t_{i-1}, t_{i+1})
            correct_cond_pred = tf.equal(tf.argmax(conditional, 2), tf.argmax(self.targets, 2))
            correct_cond_pred = tf.cast(correct_cond_pred,"float")
            cond_accuracy = tf.reduce_sum(correct_cond_pred * tf.reduce_sum(self.targets, 2)) /\
                            tf.reduce_sum(self.targets)
            self.cond_accuracy = cond_accuracy
            # log-likelihood
            pots_list = tf.split(0, config.batch_size, self.pots_layer)
            pots_list = [tf.squeeze(pots) for pots in pots_list]
            tags_list = tf.split(0, config.batch_size, self.tags)
            tags_list = [tf.squeeze(tags) for tags in tags_list]
            args_list = zip(pots_list, tags_list)
            self.args_list = args_list      # useless -> DEBUG
            dynamic = [tf.user_ops.chain_sum_product(pots, tags)
                       for pots, tags in args_list]
            self.dynamic = dynamic          # useless -> DEBUG
            pre_crf_list = [(pots, tags, f_sp, b_sp, grads)
                            for ((pots, tags), (f_sp, b_sp, grads)) in zip(args_list, dynamic)]
            crf_list = [tf.user_ops.chain_crf(pots, tags, f_sp, b_sp, grads)
                        for (pots, tags, f_sp, b_sp, grads) in pre_crf_list]
            log_likelihoods = tf.pack([ll for (ll, marg) in crf_list])
            self.log_likelihoods = log_likelihoods    # useless -> DEBUG
            self.marginals = tf.pack([marg for (ll, marg) in crf_list])
            log_likelihood = tf.reduce_sum(log_likelihoods)
            self.log_likelihood = log_likelihood
            # L1 regularization
            self.l1_norm = tf.reduce_sum(tf.zeros([1]))
            for feat in config.l1_list:
                self.l1_norm += config.l1_reg * \
                                tf.reduce_sum(tf.abs(params.embeddings[feat]))
            # L2 regularization
            self.l2_norm = tf.reduce_sum(tf.zeros([1]))
            for feat in config.l2_list:
                self.l2_norm += config.l2_reg * \
                                tf.reduce_sum(tf.mul(params.embeddings[feat],
                                                     params.embeddings[feat]))
            # map assignment and accuracy of map assignment
            map_tags = map_assignment(self.pots_layer, config)
            correct_pred = tf.equal(map_tags, tf.argmax(self.targets, 2))
            correct_pred = tf.cast(correct_pred,"float")
            accuracy = tf.reduce_sum(correct_pred * tf.reduce_sum(self.targets, 2)) /\
                       tf.reduce_sum(self.targets)
            self.map_tags = map_tags
            self.accuracy = accuracy
            # different criteria
            self.criteria = {}
            self.criteria['pseudo_ll'] = -self.pseudo_ll
            self.criteria['likelihood'] = -self.log_likelihood
            norm_penalty = config.l1_reg * self.l1_norm + config.l1_reg * self.l2_norm
            for k in self.criteria:
                self.criteria[k] -= norm_penalty
            # corresponding training steps, gradient clipping
            optimizers = {}
            for k in self.criteria:
                optimizers[k] = tf.train.AdagradOptimizer(config.learning_rate, name='adagrad_' + k)
            grads_and_vars = {}
            for k, crit in self.criteria.items():
                uncapped_g_v = optimizers[k].compute_gradients(crit,
                                                               tf.trainable_variables())
                print config.gradient_clip
                grads_and_vars[k] = [(tf.clip_by_norm(g, config.gradient_clip), v)
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
            batch.read(data, i * batch_size, config)
            f_dict = {self.input_ids: batch.features,
                      self.pot_indices: batch.tag_neighbours_lin,
                      self.window_indices: batch.tag_windows_lin,
                      self.mask: batch.mask,
                      self.targets: batch.tags_one_hot,
                      self.tags: batch.tags}
            if (i == 0):
                print('First crit: %f' % (criterion.eval(feed_dict=f_dict),))
            train_step.run(feed_dict=f_dict)
            crit = criterion.eval(feed_dict=f_dict)
            total_crit += crit
            if i % 50 == 0:
                train_accuracy = self.accuracy.eval(feed_dict=f_dict)
                print("step %d of %d, training accuracy %f, criterion %f" %
                      (i, n_batches, train_accuracy, crit))
        print 'total crit', total_crit / n_batches
        return total_crit / n_batches
    
    def validate_accuracy(self, data, config):
        batch_size = config.batch_size
        batch = Batch()
        total_accuracy = 0.
        total_cond_accuracy = 0.
        total_ll = 0.
        total_pll = 0.
        total = 0.
        for i in range(len(data) / batch_size):
            batch.read(data, i * batch_size, config)
            f_dict = {self.input_ids: batch.features,
                      self.pot_indices: batch.tag_neighbours_lin,
                      self.window_indices: batch.tag_windows_lin,
                      self.mask: batch.mask,
                      self.targets: batch.tags_one_hot,
                      self.tags: batch.tags}
            dev_accuracy = self.accuracy.eval(feed_dict=f_dict)
            dev_cond_accuracy = self.cond_accuracy.eval(feed_dict=f_dict)
            pll = self.pseudo_ll.eval(feed_dict=f_dict)
            ll = self.log_likelihood.eval(feed_dict=f_dict)
            total_accuracy += dev_accuracy
            total_cond_accuracy += dev_cond_accuracy
            total_pll += pll
            total_ll += ll
            total += 1
            if i % 100 == 0:
                print("%d of %d: \t map acc: %f \t cond acc: %f \
                       \t pseudo_ll:  %f  ll:  %f" % (i, len(data) / batch_size,
                                                total_accuracy / total,
                                                total_cond_accuracy / total,
                                                total_pll / total,
                                                total_ll / total))
        return (total_accuracy / total, total_cond_accuracy / total)

