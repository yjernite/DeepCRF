from random import shuffle

from utils import *
from model_defs import *


###############################################
# NN usage functions                          #
###############################################
def train_epoch(data, inputs, targets, train_step, accuracy, config, params):
    batch_size = int(inputs.get_shape()[0])
    n_outcomes = int(targets.get_shape()[2])
    for i in range(len(data) / batch_size):
        (b_feats, b_labs, b_tags, b_pot_ids) = make_batch(data, i * batch_size, config)
        f_dict = {inputs: b_feats, targets: b_labs}
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=f_dict)
            print("step %d of %d, training accuracy %f, Lemma_l1 %f" %
                  (i, len(data) / batch_size, train_accuracy,
                   tf.reduce_sum(tf.abs(params.embeddings['lemma'])).eval()))
        train_step.run(feed_dict=f_dict)


def validate_accuracy(data, inputs, targets, accuracy, config):
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


def train_model(train_data, dev_data, inputs, targets, train_step, accuracy,
                config, params, graph):
    train_data_32 = cut_and_pad(train_data, 32, config)
    dev_data_32 = cut_and_pad(dev_data, 32, config)
    accuracies = []
    preds = {}
    for i in range(config.num_epochs):
        print i
        shuffle(train_data_32)
        train_epoch(train_data_32, inputs, targets, train_step, accuracy,
                    config, params)
        train_acc = validate_accuracy(train_data_32, inputs, targets,
                                      accuracy, config)
        dev_acc = validate_accuracy(dev_data_32, inputs, targets, accuracy,
                                    config)
        accuracies += [(train_acc, dev_acc)]
        if i % config.num_predict == config.num_predict - 1:
            preds[i+1] = tag_dataset(dev_data, config, params, graph)
    return (accuracies, preds)
