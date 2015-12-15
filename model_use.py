from random import shuffle

from utils import *
from nn_defs import *
from crf_defs import *

###############################################
# model usage functions                          #
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


# for CRFs
def fuse_preds_crf(sentence, pred, config):
    res = []
    mid = config.pred_window / 2
    for tok, scores in zip(sentence, pred):
        tok_d = dict(zip(config.tag_list, list(scores[1])))
        tok_d['word'] = tok['word']
        tok_d['label'] = tok['label'].split('_')[mid]
        res += [tok_d]
    return res


# tag a full dataset TODO: ensure compatibility with SequNN class
def tag_dataset(pre_data, config, params, mod_type):
    sequ_nn_tmp = None
    crf_tmp = None
    preds_layer_output = None
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
            if mod_type == 'sequ_nn':
                sequ_nn_tmp = SequNN(config)
                sequ_nn_tmp.make(config, params, reuse=True)
            elif mod_type == 'CRF':
                crf_tmp = CRF(config)
                crf_tmp.make(config, params, reuse=True)
        if mod_type == 'sequ_nn':
            f_dict = {sequ_nn_tmp.input_ids: batch.features}
            preds_layer_output = sequ_nn_tmp.preds_layer.eval(feed_dict=f_dict)
        elif mod_type == 'CRF':
            f_dict = make_feed_crf(crf_tmp, batch)
            preds_layer_output = crf_tmp.marginals.eval(feed_dict=f_dict)
        tmp_preds = [[(batch.tag_windows_one_hot[i][j].index(1), token_preds)
                      for j, token_preds in enumerate(sentence) if 1 in batch.tag_windows_one_hot[i][j]]
                     for i, sentence in enumerate(list(preds_layer_output))]
        res += tmp_preds
    # re-order data
    res = res[:len(pre_data)]
    res = [dat for i, dat in sorted(zip(mixed_indices, res), key=lambda x:x[0])]
    config.num_steps = save_num_steps
    return res


def train_model(train_data, dev_data, model, config, params, mod_type):
    train_data_32 = cut_and_pad(train_data, config)
    dev_data_32 = cut_and_pad(dev_data, config)
    #~ train_data_32 = cut_batches(train_data, config)
    #~ dev_data_32 = cut_batches(dev_data, config)
    accuracies = []
    preds = {}
    for i in range(config.num_epochs):
        print i
        shuffle(train_data_32)
        model.train_epoch(train_data_32, config, params)
        train_acc = model.validate_accuracy(train_data_32, config)
        dev_acc = model.validate_accuracy(dev_data_32, config)
        accuracies += [(train_acc, dev_acc)]
        if i % config.num_predict == config.num_predict - 1:
            preds[i+1] = tag_dataset(dev_data, config, params, mod_type)
    return (accuracies, preds)

