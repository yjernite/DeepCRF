import numpy as np
from random import shuffle

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

from bi_rnn import bi_rnn


###############################################
# NN creation functions                       #
###############################################
class Parameters:
    def __init__(self, init={}, emb={}, w_c=False, b_c=False, w_p=False,
                 b_p=False):
        self.init_dic = init
        self.embeddings = emb
        self.W_conv = w_c
        self.b_conv = b_c
        self.W_pred = w_p
        self.b_pred = b_p


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


def feature_layer(config, params, reuse=False):
    in_features = config.in_features
    features_dim = config.features_dim
    batch_size = config.batch_size
    num_steps = config.num_steps
    feature_mappings = config.feature_maps
    # inputs
    num_features = len(in_features)
    input_ids = tf.placeholder(tf.int32, shape=[batch_size, num_steps,
                                                num_features])
    if reuse:
        tf.get_variable_scope().reuse_variables()
        param_vars = params.embeddings
    # lookup layer
    else:
        param_dic = params.init_dic
        param_vars = {}
        for feat in in_features:
            if feat in param_dic:
                param_vars[feat] = \
                  tf.Variable(tf.convert_to_tensor(param_dic[feat],
                                                   dtype=tf.float32),
                              name=feat + '_embedding',
                              trainable=False)
            else:
                shape = [len(feature_mappings[feat]['reverse']), features_dim]
                initial = tf.truncated_normal(shape, stddev=0.1)
                param_vars[feat] = tf.Variable(initial,
                                               name=feat + '_embedding')
    params = [param_vars[feat] for feat in in_features]
    input_embeddings = tf.nn.embedding_lookup(params, input_ids, name='lookup')
    # add and return
    embedding_layer = tf.reduce_sum(input_embeddings, 2)
    return (input_ids, embedding_layer, param_vars)


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
    flat_input = tf.reshape(in_layer, [-1, input_size])
    pre_scores = tf.nn.softmax(tf.matmul(flat_input, W_pred) + b_pred)
    preds_layer = tf.reshape(pre_scores, [batch_size, num_steps, -1])
    return (preds_layer, W_pred, b_pred)


def optim_outputs(outcome, config, params):
    batch_size = int(outcome.get_shape()[0])
    num_steps = int(outcome.get_shape()[1])
    n_outputs = int(outcome.get_shape()[2])
    targets = tf.placeholder(tf.float32, [batch_size, num_steps, n_outputs])
    # We are currently using cross entropy as criterion
    criterion = -tf.reduce_sum(targets * tf.log(outcome))
    for feat in config.l1_list:
        criterion += config.l1_reg * \
                     tf.reduce_sum(tf.abs(params.embeddings[feat]))
    # We also compute the per-tag accuracy
    correct_prediction = tf.equal(tf.argmax(outcome, 2), tf.argmax(targets, 2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                     "float") * tf.reduce_sum(targets, 2)) /\
        tf.reduce_sum(targets)
    return (targets, criterion, accuracy)


def make_network(config, params, reuse=False, name='Model'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        (input_ids, out_layer, embeddings) = feature_layer(config, params,
                                                           reuse=reuse)
        params.embeddings = embeddings
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
            if config.verbose:
                print('convolution layer done')
        (preds_layer, W_pred, b_pred) = predict_layer(out_layer, config,
                                                      params, reuse=reuse)
        params.W_pred = W_pred
        params.b_pred = b_pred
        (targets, criterion, accuracy) = optim_outputs(preds_layer, config,
                                                       params)
        if config.verbose:
            print('output layer done')
    return (input_ids, targets, preds_layer, criterion, accuracy)


###############################################
# NN usage functions                          #
###############################################
def train_epoch(data, inputs, targets, train_step, accuracy, config):
    input_features = config.in_features
    feature_mappings = config.feature_maps
    label_dict = config.label_dict
    batch_size = int(inputs.get_shape()[0])
    n_outcomes = int(targets.get_shape()[2])
    for i in range(len(data) / batch_size):
        (b_feats, b_labs) = make_batch(data, i * batch_size, config)
        f_dict = {inputs: b_feats, targets: b_labs}
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=f_dict)
            print("step %d of %d, training accuracy %f, Lemma_l1 %f" %
                  (i, len(data) / batch_size, train_accuracy,
                   tf.reduce_sum(tf.abs(params.embeddings['lemma'])).eval()))
        train_step.run(feed_dict=f_dict)


def validate_accuracy(data, inputs, targets, accuracy, config):
    input_features = config.in_features
    feature_mappings = config.feature_maps
    label_dict = config.label_dict
    batch_size = int(inputs.get_shape()[0])
    n_outcomes = int(targets.get_shape()[2])
    total_accuracy = 0.
    total = 0.
    for i in range(len(data) / batch_size):
        (b_feats, b_labs) = make_batch(data, i * batch_size, config)
        f_dict = {inputs: b_feats, targets: b_labs}
        dev_accuracy = accuracy.eval(feed_dict=f_dict)
        total_accuracy += dev_accuracy
        total += 1
        if i % 100 == 0:
            print("%d of %d: \t:%f" % (i, len(data) / batch_size,
                                       total_accuracy / total))
    return total_accuracy / total


# combines a sentence with the predicted marginals
def fuse_preds(sentence, pred, config):
    res = []
    mid = config.pred_window / 2
    for tok in zip(sentence, pred):
        tok_d = dict([(tag, 0) for tag in ['B', 'I', 'O', 'ID', 'OD']])
        for lab, idx in config.label_dict.items():
            tag = lab.split('_')[mid]
            if idx >= 0:
                tok_d[tag] += tok[1][1][idx]
        tok_d['word'] = tok[0]['word']
        tok_d['label'] = tok[0]['label'].split('_')[mid]
        res += [tok_d]
    return res


# tag a full dataset
def tag_dataset(pre_data, config, params, graph):
    save_num_steps = config.num_steps
    batch_size = config.batch_size
    hidden_units = config.rnn_hidden_units
    rnn_out_dim = config.rnn_output_size
    conv_window = config.conv_window
    conv_dim = config.conv_dim
    n_outcomes = config.n_outcomes
    # first, sort by length for computational reasons
    num_dev = enumerate(pre_data)
    mixed = sorted(num_dev, key=lambda x: len(x[1]))
    mixed_data = [dat for i, dat in mixed]
    mixed_indices = [i for i, dat in mixed]
    # completing the last batch
    missing = (batch_size - (len(pre_data) % batch_size)) % batch_size
    data = mixed_data + missing * [mixed_data[-1]]
    # tagging sentences
    res = []
    config.num_steps = 0
    preds_layer_s = []
    in_words = []
    print 'processing %d sentences' % ((len(data) / batch_size) * batch_size,)
    for i in range(len(data) / batch_size):
        (b_feats, b_labs) = make_batch(data, i * batch_size, config, fill=True)
        if i % 100 == 0:
            print 'making features', i, 'of', len(data) / batch_size,
            print 'rnn size', config.num_steps
        n_words = len(b_feats[0])
        if n_words > config.num_steps:
            config.num_steps = n_words
            with graph.as_default():
                with graph.device(device_for_node):
                    tf.get_variable_scope().reuse_variables()
                    (input_ids, targets, preds_layer, criterion,
                     accuracy) = make_network(config, params, reuse=True)
        f_dict = {input_ids: b_feats}
        tmp_preds = [[(b_labs[i][j].index(1), token_preds)
                      for j, token_preds in enumerate(sentence) if 1 in b_labs[i][j]]
                     for i, sentence in enumerate(list(preds_layer.eval(feed_dict=f_dict)))]
        res += tmp_preds
    # re-order data
    res = res[:len(pre_data)]
    res = [dat for i, dat in sorted(zip(mixed_indices, res), key=lambda x:x[0])]
    config.num_steps = save_num_steps
    return res


def train_model(train_data, dev_data, inputs, targets, accuracy,
                config, graph):
    train_data_32 = cut_and_pad(train_data, 32, config)
    dev_data_32 = cut_and_pad(dev_data, 32, config)
    accuracies = []
    preds = {}
    for i in range(num_epochs):
        print i
        shuffle(train_data_32)
        train_epoch(train_data_32, input_ids, targets, train_step, accuracy,
                    config)
        train_acc = validate_accuracy(train_data_32, input_ids, targets,
                                      accuracy, config)
        dev_acc = validate_accuracy(dev_data_32, input_ids, targets, accuracy,
                                    config)
        accuracies += [(train_acc, dev_acc)]
        if i % config.num_predict == config.num_predict - 1:
            preds[i+1] = tag_dataset(dev_data, config, params)
            predictions = [fuse_preds(sent, pred, config)
                           for sent, pred in zip(dev_data, preds[i+1])]
            merged = merge(predictions, dev_spans)
            for i in range(10):
                evaluate(merged, 0.1 * i)
    return (accuracies, preds)
