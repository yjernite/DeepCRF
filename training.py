from pprint import pprint

from model_config import *
from model_defs import *
from model_use import *

###############################################
# Load the data                               #
###############################################
config = base_convo_config(input_features, l1_list, tag_list)

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


###############################################
# make and test the CRF                       #
###############################################

from crf_use import *

config.pot_window = 3
config.features_dim = config.n_tags ** config.pot_window
config.l1_reg = 0

graph = tf.Graph()
with graph.as_default():
    with graph.device(device_for_node):
        sess = tf.InteractiveSession()
        (input_ids, pot_indices, targets,
            criterion, accuracy, map_tags) =  make_crf(config, params)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(criterion)
        sess.run(tf.initialize_all_variables())


train_data_32 = cut_batches(train_data, config)[:2000]
dev_data_32 = cut_batches(dev_data, config)[:2000]

for i in range(10):
    print 'epoch ----------------', i
    shuffle(train_data_32)
    train_epoch_crf(train_data_32, input_ids, targets, pot_indices, train_step, accuracy, criterion, config, params)
    validate_accuracy_crf(train_data_32, input_ids, targets, accuracy, config)
    validate_accuracy_crf(dev_data_32, input_ids, targets, accuracy, config)


###############################################
# make and test the NN                        #
###############################################

graph = tf.Graph()
with graph.as_default():
    with graph.device(device_for_node):
        sess = tf.InteractiveSession()
        (inputs, targets, preds_layer, criterion, accuracy) =  make_network(config, params)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(criterion)
        sess.run(tf.initialize_all_variables())

accuracies, preds = train_model(train_data, dev_data, inputs, targets,
                                train_step, accuracy, config, params, graph)

predictions = [fuse_preds(sent, pred, config)
               for sent, pred in zip(dev_data, preds[config.num_epochs])]

merged = merge(predictions, dev_spans)

if True:
    print '##### Parameters'
    pprint(config.to_string().splitlines())
    print '##### Train/dev accuracies'
    pprint(accuracies)
    print '##### P-R-F curves'
    for i in range(10):
        evaluate(merged, 0.1 * i)

#~ execfile('training.py')
