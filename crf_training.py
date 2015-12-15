from pprint import pprint
from model_config import *
from model_use import *
from crf_defs import *

###############################################
# Load the data                               #
###############################################
config = base_crf_config(input_features, l1_list, tag_list)

train_data = read_data(train_file, features, config)
dev_data = read_data(dev_file, features, config)
dev_spans = treat_spans(dev_spans_file)

config.make_mappings(train_data + dev_data)

if config.init_words:
    word_vectors = read_vectors(vecs_file, config.feature_maps['word']['reverse'])
    pre_trained = {'word': word_vectors}
else:
    pre_trained = {}

params_crf = Parameters(init=pre_trained)
params_nn = Parameters(init=pre_trained)

#~ train_data_32 = cut_batches(train_data, config)
#~ dev_data_32 = cut_batches(dev_data, config)

train_data_32 = cut_and_pad(train_data, config)
dev_data_32 = cut_and_pad(dev_data, config)

sess = tf.InteractiveSession()


###############################################
# make and test the CRF                       #
###############################################


### log-likelihood
config.learning_rate = 1e-3
config.l2_list = config.input_features

config.gradient_clip = 1
config.param_clip = 25

crf = CRF(config)
crf.make(config, params_crf)


sequ_nn = SequNN(config)
sequ_nn.make(config, params_nn)

sess.run(tf.initialize_all_variables())

accuracies_nn, preds_nn = train_model(train_data, dev_data, sequ_nn, config, params_nn, 'sequ_nn')
accuracies_crf, preds_crf = train_model(train_data, dev_data, crf, config, params_crf, 'CRF')


crf_predictions = [fuse_preds_crf(sent, pred, config)
                   for sent, pred in zip(dev_data, preds[config.num_epochs])]

nn_predictions = [fuse_preds(sent, pred, config)
                  for sent, pred in zip(dev_data, preds[config.num_epochs])]


def combine_toks(tok_crf, tok_nn):
    res = {}
    res['word'] = tok_nn['word']
    res['label'] = tok_nn['label']
    for tag in ['B', 'I', 'ID', 'O', 'OD']:
        res[tag] = max(tok_crf[tag], tok_nn[tag])
    return res


predictions = [[combine_toks(tok_crf, tok_nn)
                for tok_crf, tok_nn in zip(sent_crf, sent_nn)]
               for sent_crf, sent_nn in zip (crf_predictions, nn_predictions)]

merged = merge(predictions, dev_spans)

if True:
    print '##### Parameters'
    pprint(config.to_string().splitlines())
    print '##### Train/dev accuracies'
    pprint(accuracies)
    print '##### P-R-F curves'
    for i in range(10):
        evaluate(merged, 0.1 * i)

#~ execfile('crf_training.py')
