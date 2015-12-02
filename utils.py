# A few utility functions
import itertools
import numpy as np


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
    def __init__(self, batch_size=20, num_steps=32, learning_rate=1e-2,
                 l1_reg=2e-3, l1_list=[],
                 features_dim=50, init_words=False, input_features=[],
                 use_rnn=False, rnn_hidden_units=100, rnn_output_size=50,
                 use_convo=False, conv_window=5, conv_dim=50,
                 pot_window=1,
                 pred_window=1, tag_list=[],
                 verbose=False, num_epochs=10, num_predict=5):
        # optimization parameters
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        # regularization parameters
        self.l1_reg = l1_reg
        self.l1_list = l1_list
        # input layer
        self.features_dim = features_dim
        self.init_words = init_words
        self.input_features = input_features
        # recurrent layer
        self.use_rnn = use_rnn
        self.rnn_hidden_units = rnn_hidden_units
        self.rnn_output_size = rnn_output_size
        # convolutional layer
        self.use_convo = use_convo
        self.conv_window = conv_window
        self.conv_dim = conv_dim
        # CRF parameters:
        self.pot_window = pot_window
        self.n_tags = len(tag_list)
        # output layer
        self.pred_window = pred_window
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
        # tag_windows_one_hot: '<P>_B_O' -> [0, ..., 0, 1, 0, ..., 0]
        self.tag_windows_one_hot = []
        # tag_neighbours: '<P>_B_O' -> [0 , 3]
        self.tag_neighbours = []
        # tag_neighbours_linearized: num_values * token_id + '<P>_B_O' -> 0 * config.n_tags + 3
        self.tag_neighbours_lin = []
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
        self.tags_one_hot = [[[int(x == label[1])  # TODO: count padding tokens?
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
                self.tags_one_hot[i] = [[0] * config.n_outcomes] * pre_len + \
                                       self.tags_one_hot[i] + \
                                       [[0] * config.n_outcomes] * post_len
                self.tag_windows_one_hot[i] = [[0] * config.n_outcomes] * pre_len + \
                                              self.tag_windows_one_hot[i] + \
                                              [[0] * config.n_outcomes] * post_len
        mid = config.pot_window / 2
        padded_tags = [[0] * mid + sentence + [0] * mid
                       for sentence in self.tags]
        # get linearized potential indices
        self.tag_neighbours = [[sent[i + j]
                                for j in range(-mid, 0) + range(1, mid + 1)]
                               for sent in padded_tags
                               for i in range(mid, len(sent) - mid)]
        max_pow = config.pot_window - 1
        all_idx = config.n_tags ** max_pow
        self.tag_neighbours_lin = [(all_idx * i + \
                                    sum([idx * (config.n_tags) ** (max_pow - j - 1)
                                         for j, idx in enumerate(token)]))
                                   for i, token in enumerate(self.tag_neighbours)]


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


# extract windows from data to fit into unrolled RNN. Independent sentences
def cut_and_pad(data, config):
    pad_token = dict([(feat, '_unk_') for feat in data[0][0]])
    pad_token['label'] = '_'.join(['<P>'] * config.pred_window)
    num_steps = config.num_steps
    res = []
    seen = 0
    sen = [pad_token] + data[0] + [pad_token]
    while seen < len(data):
        if len(sen) < num_steps:
            if sen[0]['label'] == '<P>':
                new_sen = ((num_steps - len(sen)) / 2) * [pad_token] + sen
            else:
                new_sen = sen
            new_sen = new_sen + (num_steps - len(new_sen)) * [pad_token]
            res += [new_sen[:]]
            seen += 1
            if seen < len(data):
                sen = [pad_token] + data[seen] + [pad_token]
        else:
            res += [sen[:num_steps]]
            sen = sen[(2 * num_steps) / 3:]
    return res


# extract windows from data to fit into unrolled RNN. Continuous model
def cut_batches(data, config):
    pad_token = dict([(feat, '_unk_') for feat in data[0][0]])
    pad_token['label'] = '_'.join(['<P>'] * config.pred_window)
    padding = [pad_token] * config.pred_window
    new_data = padding + [tok for sentence in data
                          for tok in sentence + padding]
    step_size = (config.num_steps / 2)
    num_cuts = len(new_data) / step_size
    res = [new_data[i * step_size: i * step_size + config.num_steps]
           for i in range(num_cuts)]
    res[-1] = res[-1] + [pad_token] * (config.num_steps - len(res[-1]))
    return res


###############################################
# NN evaluation functions                     #
###############################################
def treat_spans(spans_file):
    span_lists = []
    f = open(spans_file)
    y = []
    for line in f:
        if line.strip() == '':
            span_lists += [y[:]]
            y = []
        else:
            lsp = line.strip().split()
            y = y + [(int(lsp[0]), int(lsp[1]), lsp[2])]
    f.close()
    return span_lists


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


def merge(sentences, spans):
    res = []
    sent = read_sentence(sentences[0])
    span = spans[0]
    for i, sp in enumerate(spans):
        if i == 0:
            continue
        if sp[0] == span[0]:
            sen = read_sentence(sentences[i])
            gold = sorted(list(set(sen[1] + sent[1])))
            sent = (sen[0], gold, sen[2])
        else:
            res += [(sent, span)]
            sent = read_sentence(sentences[i])
            span = spans[i]
    res += [(sent, span)]
    return res


def evaluate(merged_sentences, threshold):
    TP = 0
    FP = 0
    FN = 0
    for sentence in merged_sentences:
        true_mentions = sentence[0][1]
        tp = 0
        for pred in sentence[0][2]:
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
