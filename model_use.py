from random import shuffle

from utils import *
from model_defs import *


###############################################
# NN usage functions                          #
###############################################
# combines a sentence with the predicted marginals
def fuse_preds(sentence, pred, config):
    res = []
    mid = config.pred_window / 2
    for tok in zip(sentence, pred):
        tok_d = dict([(tag, 0) for tag in ['B', 'I', 'O', 'ID', 'OD']])
        for lab, idx in config.label_dict.items():
            tag = config.tag_list[idx[1]]
            if idx[0] >= 0:
                tok_d[tag] += tok[1][1][idx[0]]
        tok_d['word'] = tok[0]['word']
        tok_d['label'] = tok[0]['label'].split('_')[mid]
        res += [tok_d]
    return res


# tag a full dataset
def tag_dataset(pre_data, config, params, graph):
    save_num_steps = config.num_steps
    batch_size = config.batch_size
    batch = Batch()
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
        batch.read(data, i * batch_size, config, fill=True)
        if i % 100 == 0:
            print 'making features', i, 'of', len(data) / batch_size,
            print 'rnn size', config.num_steps
        n_words = len(batch.features[0])
        if n_words > config.num_steps:
            config.num_steps = n_words
            tf.get_variable_scope().reuse_variables()
            (input_ids, targets, preds_layer, criterion,
                  accuracy) = make_network(config, params, reuse=True)
        f_dict = {input_ids: batch.features}
        tmp_preds = [[(batch.tag_windows_one_hot[i][j].index(1), token_preds)
                      for j, token_preds in enumerate(sentence) if 1 in batch.tag_windows_one_hot[i][j]]
                     for i, sentence in enumerate(list(preds_layer.eval(feed_dict=f_dict)))]
        res += tmp_preds
    # re-order data
    res = res[:len(pre_data)]
    res = [dat for i, dat in sorted(zip(mixed_indices, res), key=lambda x:x[0])]
    config.num_steps = save_num_steps
    return res


def train_model(train_data, dev_data, sequ_nn, config, params, graph):
    #~ train_data_32 = cut_and_pad(train_data, config)
    #~ dev_data_32 = cut_and_pad(dev_data, config)
    train_data_32 = cut_batches(train_data, config)
    dev_data_32 = cut_batches(dev_data, config)
    accuracies = []
    preds = {}
    for i in range(config.num_epochs):
        print i
        shuffle(train_data_32)
        sequ_nn.train_epoch(train_data_32, config, params)
        train_acc = sequ_nn.validate_accuracy(train_data_32, config)
        dev_acc = sequ_nn.validate_accuracy(dev_data_32, config)
        accuracies += [(train_acc, dev_acc)]
        if i % config.num_predict == config.num_predict - 1:
            preds[i+1] = tag_dataset(dev_data, config, params, graph)
    return (accuracies, preds)
