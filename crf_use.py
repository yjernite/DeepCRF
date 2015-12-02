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
        pot_indices = tf.placeholder(tf.int32, [batch_size * num_steps])
        # targets <- batch.tags_one_hot
        targets = tf.placeholder(tf.float32, [batch_size, num_steps, config.n_tags])
        pseudo_ll = pseudo_likelihood(potentials, pots_indices, target,
                                      config)
        criterion = pseudo_ll
        # L1 regularization
        for feat in config.l1_list:
            criterion += config.l1_reg * \
                         tf.reduce_sum(tf.abs(params.embeddings[feat]))
        # compute map accuracy
        map_tags = map_assignment(potentials, config)
        correct_prediction = tf.equal(map_tags, tf.argmax(targets, 2))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                 "float") * tf.reduce_sum(targets, 2)) /\
                  tf.reduce_sum(targets)
        # TODO: add marginal inference
    return (input_ids, pots_indices, targets, criterion, accuracy, map_tags)
