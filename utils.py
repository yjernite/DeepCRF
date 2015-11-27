# A few utility functions
import itertools
import numpy as np


###############################################
# Data reading functions                      #
###############################################
def valid_tag(tag_tup):
    ln = len(tag_tup)
    mid = ln / 2
    if '<P>' in tag_tup:
        for i, t in enumerate(tag_tup[1:-1]):
            if t == '<P>':
                if tag_tup[mid] == '<P>':
                    return False
                else:
                    ct_a = tag_tup[:mid].count('<P>')
                    ct_b = tag_tup[mid + 1:].count('<P>')
                    if ((ct_a == mid and ct_b == 0) or
                       (ct_b == mid and ct_a == 0)):
                        return True
                return False
    return True


class Config:
    def __init__(self, batch_size=20, num_steps=32, learning_rate=1e-2,
                 l1_reg=2e-3, l1_list=[],
                 features_dim=50, init_words=False, input_features=[],
                 use_rnn=False, rnn_hidden_units=100, rnn_output_size=50,
                 use_convo=False, conv_window=5, conv_dim=50,
                 pred_window=1, tag_list=[],
                 verbose=False, num_epochs=20, num_predict=5):
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
        # output layer
        self.pred_window = pred_window
        self.tag_list = tag_list
        self.label_dict = {}
        tags_ct = 0
        for element in itertools.product(tag_list, repeat=pred_window):
            tag_st = '_'.join(element)
            if valid_tag(element):
                self.label_dict[tag_st] = tags_ct
                tags_ct += 1
            else:
                self.label_dict[tag_st] = -1
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


def aggregate_labels(sentence, config):
    pre_tags = ['P'] * (config.pred_window / 2)
    sentence_ext = pre_tags + [token['label']
                               for token in sentence] + pre_tags
    for i, token in enumerate(sentence):
        current = token['label']
        sentence[i]['label'] = '_'.join([sentence_ext[i+j]
                                         for j in range(config.pred_window)])


def read_data(file_name, features, config):
    gc.disable()
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
    gc.enable()
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


# extract windows from data to fit into unrolled RNN
def cut_and_pad(data, num_steps, config):
    pad_token = dict([(feat, '_unk_') for feat in data[0][0]])
    pad_token['label'] = '_'.join(['P'] * config.pred_window)
    res = []
    seen = 0
    sen = [pad_token] + data[0] + [pad_token]
    while seen < len(data):
        if len(sen) < num_steps:
            if sen[0]['label'] == 'P':
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


# make a batch: batch_size x num_steps x num_features
def make_batch(data, start, config, fill=False):
    batch_size = config.batch_size
    features_list = config.in_features
    n_outcomes = config.n_outcomes
    feature_mappings = config.feature_maps
    label_dict = config.label_dict
    num_features = len(features_list)
    batch_data = data[start:start + batch_size]
    batch_features = [[[feature_mappings[feat]['lookup'][token[feat]]
                        for feat in features_list]
                       for token in sentence]
                      for sentence in batch_data]
    batch_labels = [[label_dict[token['label']] for token in sentence]
                    for sentence in batch_data]
    b_feats = [[[num_features * ft + i for i, ft in enumerate(word)]
               for word in sentence] for sentence in batch_features]
    b_labs = [[[int(x == label) for x in range(n_outcomes)]
               for label in sentence]
              for sentence in batch_labels]
    if fill:
        max_len = max(config.conv_window,
                      max([len(sentence) for sentence in batch_data]) + 2)
        for i in range(batch_size):
            current_len = len(b_feats[i])
            pre_len = (max_len - current_len) / 2
            post_len = max_len - pre_len - current_len
            b_feats[i] = [range(num_features)] * pre_len + b_feats[i] + \
                         [range(num_features)] * post_len
            b_labs[i] = [[0] * n_outcomes] * pre_len + b_labs[i] + \
                        [[0] * n_outcomes] * post_len
    return (b_feats, b_labs)


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
