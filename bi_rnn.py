from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import control_flow_ops


def bi_rnn(cell_forward, cell_backward, inputs, initial_state=None,
           dtype=None, scope=None, reuse=False):
    if not (isinstance(cell_forward, rnn_cell.RNNCell) and
            isinstance(cell_backward, rnn_cell.RNNCell)):
        raise TypeError("cell must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")
    outputs = []
    states = []
    with tf.variable_scope(scope or "RNN"):
        batch_size = tf.shape(inputs[0])[0]
        outputs_f = [0] * len(inputs)
        states_f = [0] * len(inputs)
        outputs_b = [0] * len(inputs)
        states_b = [0] * len(inputs)
        if initial_state is not None:
            state_f = initial_state
            state_b = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, \
                                  dtype must be.")
            state_f = cell_forward.zero_state(batch_size, dtype)
            state_b = cell_backward.zero_state(batch_size, dtype)
        for t, input_ in enumerate(inputs):
            if reuse or t > 0:
                tf.get_variable_scope().reuse_variables()
            output_f, state_f = cell_forward(inputs[t], state_f,
                                             scope='LSTM_f')
            output_b, state_b = cell_backward(inputs[-1 - t], state_b,
                                              scope='LSTM_b')
            outputs_f[t] = output_f
            outputs_b[-1 - t] = output_b
            states_f[t] = state_f
            states_b[-1 - t] = state_b
        for t in range(len(inputs)):
            outputs.append(tf.concat(1, [outputs_f[t], outputs_b[t]]))
            states.append(tf.concat(1, [states_f[t], states_b[t]]))
    return (outputs, states)
