# A few utility functions
import itertools
import numpy as np
import tensorflow as tf
from pprint import pprint
from random import shuffle


###############################################
# Generally useful functions                  #
###############################################
# useful with reshape
def linearize_indices(indices, dims):
    res = []
    remain = indices
    for i, _ in enumerate(dims):
        res = [remain % dims[-i - 1]] + res
        remain = remain / dims[-i - 1]
    linearized = tf.transpose(tf.pack(res))
    return linearized


###############################################
# Data reading functions                      #
###############################################
class Config:
    def __init__(self, batch_size=20, learning_rate=1e-2,
                 l1_reg=1e-2, l1_list=[], l2_reg=1e-2, l2_list=[],
                 nn_obj_weight=-1,
                 optimizer='adam', criterion='likelihood',
                 gradient_clip=1e0, param_clip=1e2,
                 features_dim=200, init_words=False,
                 input_features=[], combine='sum',
                 use_convo=True, bi_convo=False, conv_window=5, conv_dim=200,
                 pot_size=1,
                 pred_window=1, tag_list=[],
                 verbose=False, num_epochs=10, num_predict=5):
        # optimization parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # regularization parameters
        self.l1_reg = l1_reg
        self.l1_list = l1_list
        self.l2_reg = l2_reg
        self.l2_list = l2_list
        self.nn_obj_weight = nn_obj_weight  # for mixed training
        # optimization configuration
        self.optimizer = optimizer          # ['adam', 'adagrad']
        self.criterion = criterion          # ['likelihood', 'pseudo_ll']
        self.gradient_clip = gradient_clip
        self.param_clip = param_clip
        # input layer
        self.features_dim = features_dim
        self.init_words = init_words
        self.input_features = input_features
        self.combine = combine
        # convolutional layer
        self.use_convo = use_convo
        self.bi_convo = bi_convo
        self.conv_window = conv_window
        self.conv_dim = conv_dim
        # CRF parameters:
        self.pot_size = pot_size
        self.n_tags = len(tag_list)
        # output layer
        self.pred_window = pred_window
        self.n_tag_windows = self.n_tags ** pred_window
        self.tag_list = tag_list
        self.label_dict = {}
        tags_ct = 0
        for element in itertools.product(tag_list, repeat=pred_window):
            tag_st = '_'.join(element)
            mid = element[pred_window / 2]
            if mid == '<P>':
                self.label_dict[tag_st] = (-1, tag_list.index(mid))
            else:
                self.label_dict[tag_st] = (tags_ct, tag_list.index(mid))
            tags_ct += 1
        self.n_outcomes = tags_ct
        # misc parameters
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.num_predict = num_predict

    def make_mappings(self, data):
        self.feature_maps = dict([(feat, {'lookup': {'_unk_': 0},
                                          'reverse': ['_unk_']})
                                  for feat in data[0][0]])
        for sentence in data:
            for token in sentence:
                for feat in data[0][0]:
                    ft = token[feat]
                    if ft not in self.feature_maps[feat]['lookup']:
                        self.feature_maps[feat]['lookup'][ft] = \
                                    len(self.feature_maps[feat]['reverse'])
                        self.feature_maps[feat]['reverse'] += [ft]

    def to_string(self):
        st = ''
        for k, v in self.__dict__.items():
            if k not in ['feature_maps', 'label_dict']:
                st += k + ' --- ' + str(v) + ' \n'
        return st


class Batch:
    def __init__(self):
        # features: {'word': 'have', 'pos': 'VB', ...} ->
        #                              [1345, 12 * num_features + 1,...]
        self.features = []
        # tags: 'B' -> 1
        self.tags = []
        # tags_one_hot: 'B' -> [0, 1, 0, 0, 0, 0]
        self.tags_one_hot = []
        # tag_windows: '<P>_B_O' -> [0, 1, 3]
        self.tag_windows = []
        # tag_windows_lin: '<P>_B_O' -> num_values * token_id + 0 * config.n_tags **2 + 1 * config.n_tags + 3
        self.tag_windows_lin = []
        # tag_windows_one_hot: '<P>_B_O' -> [0, ..., 0, 1, 0, ..., 0]
        self.tag_windows_one_hot = []
        # tag_neighbours: '<P>_B_O' -> [0, 3]
        self.tag_neighbours = []
        # tag_neighbours_linearized: '<P>_B_O' -> num_values * token_id + 0 * config.n_tags + 3
        self.tag_neighbours_lin = []
        # mask: <P> -> 0, everything else -> 1
    def read(self, data, start, config, fill=False):
        num_features = len(config.input_features)
        batch_data = data[start:start + config.batch_size]
        batch_features = [[[config.feature_maps[feat]['lookup'][token[feat]]
                            for feat in config.input_features]
                           for token in sentence]
                          for sentence in batch_data]
        batch_labels = [[config.label_dict[token['label']]
                         for token in sentence]
                        for sentence in batch_data]
        # multiply feature indices for use in tf.nn.embedding_lookup
        self.features = [[[num_features * ft + i for i, ft in enumerate(word)]
                         for word in sentence] for sentence in batch_features]
        self.tags = [[label[1] for label in sentence]
                     for sentence in batch_labels]
        self.tags_one_hot = [[[int(x == label[1] and x > 0)  # TODO: count padding tokens?
                               for x in range(config.n_tags)]
                              for label in sentence]
                             for sentence in batch_labels]
        self.tag_windows_one_hot = [[[int(x == label[0])
                                      for x in range(config.n_outcomes)]
                                     for label in sentence]
                                    for sentence in batch_labels]
        if fill:
            max_len = max(config.conv_window,
                          max([len(sentence) for sentence in batch_data]) + 2)
            for i in range(config.batch_size):
                current_len = len(batch_data[i])
                pre_len = (max_len - current_len) / 2
                post_len = max_len - pre_len - current_len
                self.features[i] = [range(num_features)] * pre_len + \
                                   self.features[i] + \
                                   [range(num_features)] * post_len
                self.tags[i] = [0] * pre_len + self.tags[i] + [0] * post_len
                self.tags_one_hot[i] = [[0] * config.n_tags] * pre_len + \
                                       self.tags_one_hot[i] + \
                                       [[0] * config.n_tags] * post_len
                self.tag_windows_one_hot[i] = [[0] * config.n_outcomes] * pre_len + \
                                              self.tag_windows_one_hot[i] + \
                                              [[0] * config.n_outcomes] * post_len
        mid = config.pot_size - 1
        padded_tags = [[0] * mid + sentence + [0] * mid
                       for sentence in self.tags]
        # get linearized window indices
        self.tag_windows = [[sent[i + j] for j in range(-mid, mid + 1)]
                            for sent in padded_tags
                            for i in range(mid, len(sent) - mid)]
        n_indices = config.n_tags ** (config.pot_size + 1)
        self.tag_windows_lin = [sum([t * (config.n_tags ** (config.pot_size - i))
                                      for i, t in enumerate(window)]) + i * n_indices
                                for i, window in enumerate(self.tag_windows)]
        # get linearized potential indices
        self.tag_neighbours = [[sent[i + j]
                                for j in range(-mid, 0) + range(1, mid + 1)]
                               for sent in padded_tags
                               for i in range(mid, len(sent) - mid)]
        max_pow = config.pot_size
        n_indices = config.n_tags ** max_pow
        self.tag_neighbours_lin = [sum([idx * (config.n_tags) ** (max_pow - j - 1)
                                        for j, idx in enumerate(token)]) + i * n_indices
                                   for i, token in enumerate(self.tag_neighbours)]
        # make mask:
        self.mask = [[int(tag > 0) for tag in sent] for sent in self.tags]


class Parameters:
    def __init__(self, init={}, emb={}, w_c=False, b_c=False, w_p=False,
                 b_p=False, w_po=False, b_po=False, w_po_b=False, b_po_b=False,
                 w_po_u=False, b_po_u=False):
        self.init_dic = init
        self.embeddings = emb
        self.W_conv = w_c
        self.b_conv = b_c
        self.W_pred = w_p
        self.b_pred = b_p
        self.W_pot = w_po
        self.b_pot = b_po
        self.W_pot_bin = w_po_b
        self.b_pot_bin = b_po_b
        self.W_pot_un = w_po_u
        self.b_pot_un = b_po_u


def aggregate_labels(sentence, config):
    pre_tags = ['<P>'] * (config.pred_window / 2)
    sentence_ext = pre_tags + [token['label']
                               for token in sentence] + pre_tags
    for i, token in enumerate(sentence):
        current = token['label']
        sentence[i]['label'] = '_'.join([sentence_ext[i+j]
                                         for j in range(config.pred_window)])


def read_data(file_name, features, config):
    sentences = []
    sentence = []
    f = open(file_name)
    c = 0
    for line in f:
        c += 1
        if c % 100000 == 0:
            print c, 'lines read'
        if len(line.strip()) == 0 and len(sentence) > 0:
            sentences += [sentence[:]]
            sentence = []
        else:
            sentence += [dict(zip(features, line.strip().split('\t')))]
    if len(sentence) > 0:
        sentences += [sentence[:]]
    f.close()
    foo = [aggregate_labels(sentence, config) for sentence in sentences]
    return sentences


def show(sentence):
    return ' '.join([token['word']+'/'+token['label'] for token in sentence])


# read pre_trained word vectors
def read_vectors(file_name, vocab):
    vectors = {}
    f = open(file_name)
    dim = int(f.readline().strip().split()[1])
    for line in f:
        w = line.split()[0]
        vec = [float(x) for x in line.strip().split()[1:]]
        vectors[w] = np.array(vec)
    f.close()
    res = np.zeros((len(vocab), dim))
    for i, w in enumerate(vocab):
        res[i] = vectors.get(w, np.zeros(dim))
    return res


# norm functions
def L1_norm(tensor):
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.mul(tensor, tensor), 1)))
    #return tf.reduce_sum(tf.abs(tensor))


def L2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.mul(tensor, tensor)))


###############################################
# Model use functions                         #
###############################################
# making a feed dictionary:
def make_feed_crf(model, batch):
    f_dict = {model.input_ids: batch.features,
              model.pot_indices: batch.tag_neighbours_lin,
              model.window_indices: batch.tag_windows_lin,
              model.mask: batch.mask,
              model.targets: batch.tags_one_hot,
              model.tags: batch.tags,
              model.nn_targets: batch.tag_windows_one_hot}
    return f_dict


# tag a full dataset TODO: ensure compatibility with SequNN class
def tag_dataset(pre_data, config, params, mod_type, model):
    preds_layer_output = None
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
    print 'processing %d sentences' % ((len(data) / batch_size) * batch_size,)
    for i in range(len(data) / batch_size):
        batch.read(data, i * batch_size, config, fill=True)
        if i % 100 == 0 and config.verbose:
            print 'making features', i, 'of', len(data) / batch_size
        n_words = len(batch.features[0])
        if mod_type == 'sequ_nn':
            f_dict = {sequ_nn_tmp.input_ids: batch.features}
            preds_layer_output = sequ_nn_tmp.preds_layer.eval(feed_dict=f_dict)
        elif mod_type == 'CRF':
            f_dict = make_feed_crf(model, batch)
            preds_layer_output = tf.argmax(model.map_tagging, 2).eval(feed_dict=f_dict)
        tmp_preds = [[(batch.tags[i][j], token_preds)
                      for j, token_preds in enumerate(sentence) if 1 in batch.tag_windows_one_hot[i][j]]
                     for i, sentence in enumerate(list(preds_layer_output))]
        res += tmp_preds
    # re-order data
    res = res[:len(pre_data)]
    res = [dat for i, dat in sorted(zip(mixed_indices, res), key=lambda x:x[0])]
    return res


# make nice batches
def prepare_data(data, config):
    batch_size = config.batch_size
    missing = (batch_size - (len(data) % batch_size)) % batch_size
    data_shuff = data[:] + data[-missing:]
    shuffle(data_shuff)
    data_shuff = sorted(data_shuff, key=len)
    data_batches = [data_shuff[i * batch_size: (i + 1) * batch_size]
                          for i in range(len(data_shuff) / batch_size)]
    shuffle(data_batches)
    data_ready = [x for batch in data_batches for x in batch]
    return data_ready


def train_model(train_data, dev_data, model, config, params, mod_type):
    #~ train_data_32 = cut_and_pad(train_data, config)
    #~ dev_data_32 = cut_and_pad(dev_data, config)
    accuracies = []
    preds = {}
    for i in range(config.num_epochs):
        print i
        train_data_ready = prepare_data(train_data, config)
        dev_data_ready = prepare_data(dev_data, config)
        model.train_epoch(train_data_ready, config, params)
        train_acc = model.validate_accuracy(train_data_ready, config)
        dev_acc = model.validate_accuracy(dev_data_ready, config)
        accuracies += [(train_acc, dev_acc)]
        if i % config.num_predict == config.num_predict - 1:
            preds[i+1] = tag_dataset(dev_data, config, params, mod_type, model)
    return (accuracies, preds)


###############################################
# NN evaluation functions                     #
###############################################
def find_gold(sentence):
    gold = []
    current_gold = []
    for i, token in enumerate(sentence):
        if token['label'] == 'B' or token['label'] == 'O':
            if len(current_gold) > 0:
                gold += [tuple(current_gold)]
                current_gold = []
        if 'I' in token['label'] or token['label'] == 'B':
            current_gold += [i]
    if len(current_gold) > 0:
        gold += [tuple(current_gold)]
    return gold


def make_scores(token, thr):
    res = dict([(key, val)
                for key, val in token.items()
                if key in ['O', 'OD', 'I', 'ID', 'B'] and val > thr])
    return res


def find_mentions(sentence, thr=0.02):
    scores = [make_scores(token, thr) for token in sentence]
    found = []
    working = []
    for i, score in enumerate(scores):
        if 'B' in score or 'O' in score:
            for work in working:
                if work[0][-1] == i-1:
                    sc = work[1] + np.log(score.get('B', 0) +
                                          score.get('O', 0))
                    sc /= (work[0][-1] + 2 - work[0][0])
                    found += [(tuple(work[0]), np.exp(sc))]
        if len(score) == 1 and 'O' in score:
            working = []
        else:
            new_working = []
            if 'B' in score:
                new_working = [[[i], np.log(score['B']), False]]
            for work in working:
                for tg, sc in score.items():
                    if tg == 'OD':
                        new_working += [[work[0], work[1] + np.log(sc), True]]
                    elif tg == 'ID' and work[2]:
                        new_working += [[work[0] + [i], work[1] + np.log(sc),
                                         True]]
                    elif tg == 'I' and not work[2]:
                        new_working += [[work[0] + [i], work[1] + np.log(sc),
                                         False]]
            working = new_working[:]
            if len(working) > 1000:
                working = sorted(working, key=lambda x: x[1],
                                 reverse=True)[:1000]
    return sorted(found, key=lambda x: x[1], reverse=True)


def read_sentence(sentence):
    return (sentence, find_gold(sentence), find_mentions(sentence))


def tags_to_mentions(tagging):
    rebuild = []
    core = []
    added = []
    for i, tag in enumerate(tagging):
        if tag == 'Bp':
            if len(core) > 0:
                if len(added) <= 1:
                    added += [[]]
                for a in added:
                        rebuild += [sorted(a[:] + core[:])]
                core = []
                added = []
            added += [[i]]
        if tag in ['In', 'IDn']:
            added += [[i]]
        if tag in ['Ip', 'IDp']:
            if len(added) == 0:
                added = [[]]
            added[-1] += [i]
        if tag in ['B', 'O'] and len(core) > 0:
            if len(added) <= 1:
                added += [[]]
            for a in added:
                    rebuild += [sorted(a[:] + core[:])]
            core = []
            added = []
        if tag in ['B', 'I', 'ID']:
            core += [i]
    if len(core) > 0:
        if len(added) <= 1:
            added += [[]]
        for a in added:
                rebuild += [sorted(a[:] + core[:])]
    return sorted([tuple(x) for x in rebuild])


def preds_to_sentences(model_preds, config):
    res = []
    for pred in model_preds:
        found = tags_to_mentions([config.tag_list[x[1]] for x in pred])
        gold = tags_to_mentions([config.tag_list[x[0]] for x in pred])
        res += [('', gold, tuple([(x, 1) for x in found]))]
    return res


def evaluate(sentences, threshold):
    TP = 0
    FP = 0
    FN = 0
    for sentence in sentences:
        true_mentions = sentence[1]
        tp = 0
        for pred in sentence[2]:
            if pred[1] >= threshold:
                if pred[0] in true_mentions:
                    tp += 1
                else:
                    FP += 1
        TP += tp
        FN += len(true_mentions) - tp
    if (TP + FP) == 0:
        prec = 0
        recall = 0
    else:
        prec = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
    if prec == 0 or recall == 0:
        f1 = 0
    else:
        f1 =  2 * (prec * recall) / (prec + recall)
    print 'TH:', threshold, '\t', 'P:', prec, '\t', 'R:', recall, '\t', 'F:', f1

