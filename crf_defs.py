from model_Defs import *

# takes features and outputs potentials
def potentials_layer(in_layer, config, params, reuse=False, name='Potentials'):
    pot_size = config.n_tags ** config.pot_window
    out_shape = [batch_size, num_steps] + [config.n_tags] * config.pot_window
    batch_size = int(in_layer.get_shape()[0])
    num_steps = int(in_layer.get_shape()[1])
    input_size = int(in_layer.get_shape()[2])
    if reuse:
        tf.get_variable_scope().reuse_variables()
        W_pot = params.W_pot
        b_pot = params.b_pot
    else:
        W_pot = weight_variable([input_size, pot_size], name=name)
        b_pot = bias_variable([pot_size], name=name)
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.matmul(flat_input, W_pot) + b_pot
    pots_layer = tf.reshape(pre_scores, [batch_size, num_steps, out_shape])
    return (pots_layer, W_pot, b_pot)


def pseudo_likelihood(potentials, tags, config, params):
    batch_size = int(potentials.get_shape()[0])
    num_steps = int(potentials.get_shape()[1])
    pots_shape = int(potentials.get_shape()[2:])


def pseudo_likelihood(potentials, targets, config, params):
    batch_size = int(potentials.get_shape()[0])
    num_steps = int(potentials.get_shape()[1])
    pots_shape = int(potentials.get_shape()[2:])


def map_tags(potentials, config, params):
    batch_size = int(potentials.get_shape()[0])
    num_steps = int(potentials.get_shape()[1])
    pots_shape = int(potentials.get_shape()[2:])


def marginals(potentials, config, params):
    batch_size = int(potentials.get_shape()[0])
    num_steps = int(potentials.get_shape()[1])
    pots_shape = int(potentials.get_shape()[2:])

