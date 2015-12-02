from crf_defs import *


###################################
# Making a (deep) CRF             #
###################################
def make_crf(config, params, reuse=False, name='CRF'):
    with tf.variable_scope(name):
        # input_ids <- batch.features
        if reuse:
            tf.get_variable_scope().reuse_variables()
        (input_ids, out_layer, embeddings) = feature_layer(config, params,
                                                           reuse=reuse)
        params.embeddings = embeddings
        if config.verbose:
            print('features layer done')
        # TODO: add layers
        (pots_layer, W_pot, b_pot) = potentials_layer(out_layer, config,
                                                      params, reuse=reuse)
        params.W_pot = W_pot
        params.b_pot = b_pot
        if config.verbose:
            print('potentials layer done')
        # TODO: switch criteria
        # pot_indices <- batch.tag_neighbours_lin
        pot_indices = tf.placeholder(tf.int32,
                                     [config.batch_size * config.num_steps])
        # targets <- batch.tags_one_hot
        targets = tf.placeholder(tf.float32, [config.batch_size,
                                              config.num_steps,
                                              config.n_tags])
        pseudo_ll = pseudo_likelihood(pots_layer, pot_indices, targets,
                                      config)
        criterion = pseudo_ll
        # L1 regularization
        for feat in config.l1_list:
            criterion += config.l1_reg * \
                         tf.reduce_sum(tf.abs(params.embeddings[feat]))
        # compute map accuracy
        map_tags = map_assignment(pots_layer, config)
        correct_prediction = tf.equal(map_tags, tf.argmax(targets, 2))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                 "float") * tf.reduce_sum(targets, 2)) /\
                  tf.reduce_sum(targets)
        # TODO: add marginal inference
    return (input_ids, pot_indices, targets, criterion, accuracy, map_tags)


def train_epoch_crf(data, inputs, targets, pot_indices, train_step, accuracy, criterion, config, params):
    batch_size = int(inputs.get_shape()[0])
    n_outcomes = int(targets.get_shape()[2])
    total_crit = 0.
    n_batches = len(data) / batch_size
    batch = Batch()
    for i in range(n_batches):
        batch.read(data, i * batch_size, config)
        f_dict = {inputs: batch.features, targets: batch.tags_one_hot,
                  pot_indices: batch.tag_neighbours_lin}
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=f_dict)
            print("step %d of %d, training accuracy %f, Lemma_l1 %f" %
                  (i, n_batches, train_accuracy,
                   tf.reduce_sum(tf.abs(params.embeddings['lemma'])).eval()))
        train_step.run(feed_dict=f_dict)
        total_crit += criterion.eval(feed_dict=f_dict)
    print 'total crit', total_crit / n_batches
    return total_crit / n_batches



def validate_accuracy_crf(data, inputs, targets, accuracy, config):
    batch_size = int(inputs.get_shape()[0])
    n_outcomes = int(targets.get_shape()[2])
    batch = Batch()
    total_accuracy = 0.
    total = 0.
    for i in range(len(data) / batch_size):
        batch.read(data, i * batch_size, config)
        f_dict = {inputs: batch.features, targets: batch.tags_one_hot}
        dev_accuracy = accuracy.eval(feed_dict=f_dict)
        total_accuracy += dev_accuracy
        total += 1
        if i % 100 == 0:
            print("%d of %d: \t:%f" % (i, len(data) / batch_size,
                                       total_accuracy / total))
    return total_accuracy / total


