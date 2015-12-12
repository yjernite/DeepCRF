from pprint import pprint
from random import shuffle

from model_config import *
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

params = Parameters(init=pre_trained)


#~ train_data_32 = cut_batches(train_data, config)
#~ dev_data_32 = cut_batches(dev_data, config)

train_data_32 = cut_and_pad(train_data, config)
dev_data_32 = cut_and_pad(dev_data, config)

sess = tf.InteractiveSession()


###############################################
# make and test the CRF                       #
###############################################


### log-likelihood
config.learning_rate = 1e-2
config.l1_reg = 1e-2
config.l2_list = config.input_features
config.l2_reg = 1e-2

config.gradient_clip = 1

config.use_convo = True
config.features_dim = 200

crf = CRF(config)
crf.make(config, params)
sess.run(tf.initialize_all_variables())

for i in range(5):
    print 'epoch ----------------', i
    shuffle(train_data_32)
    crf.train_epoch(train_data_32, config, params, sess)
    # crf.validate_accuracy(train_data_32, config)
    crf.validate_accuracy(dev_data_32[:4000], config)


### pseudo_ll
config.learning_rate = 1e-2
config.l1_reg = 0
config.l2_list = config.input_features
config.l2_reg = 1e-2

config.gradient_clip = 1000
config.use_convo = True
config.conv_dim = 200

crf = CRF(config)
crf.make(config, params)
sess.run(tf.initialize_all_variables())

for i in range(5):
    print 'epoch ----------------', i
    shuffle(train_data_32)
    crf.train_epoch(train_data_32, config, params, sess, crit_type='pseudo_ll')
    # crf.validate_accuracy(train_data_32, config)
    crf.validate_accuracy(dev_data_32[:4000], config)


