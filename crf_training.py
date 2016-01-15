from pprint import pprint
import os
import argparse
from datetime import datetime

from crf_defs import *

config_file = 'Configs/my_config.py'
config = None

train_file = ''
dev_file = ''
features = []

def main():
    # load the data
    train_data = read_data(train_file, features, config)
    dev_data = read_data(dev_file, features, config)
    config.make_mappings(train_data + dev_data)
    # initialize the parameters
    if config.init_words:
        word_vectors = read_vectors(vecs_file,
                                    config.feature_maps['word']['reverse'])
        pre_trained = {'word': word_vectors}
    else:
        pre_trained = {}
    params_crf = Parameters(init=pre_trained)
    # make the CRF and SequNN models
    sess = tf.InteractiveSession()
    crf = CRF(config)
    crf.make(config, params_crf)
    sess.run(tf.initialize_all_variables())
    # (accuracies, preds) = train_model(train_data, dev_data, crf, config, params_crf, 'CRF')
    for i in range(100):
        print i
        train_data_ready = prepare_data(train_data, config)
        dev_data_ready = prepare_data(dev_data, config)
        print 'training', i, '\t', str(datetime.now())
        crf.train_epoch(train_data_ready, config, params_crf)
        print 'validating', i, '\t', str(datetime.now())
        train_acc = crf.validate_accuracy(train_data_ready, config)
        dev_acc = crf.validate_accuracy(dev_data_ready, config)
        print 'train_acc', train_acc, 'dev_acc', dev_acc
        print 'tagging', i, '\t', str(datetime.now())
        preds = tag_dataset(dev_data, config, params_crf, 'CRF', crf)
        sentences = preds_to_sentences(preds, config)
        print 'epoch', i, '\t', str(datetime.now())
        evaluate(sentences, 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing the models for \
                                     various parameter values')
    parser.add_argument("-conf", "--config_file",
                        help="location of configuration file")
    args = parser.parse_args()
    if args.config_file:
        config_file = os.path.abspath(args.config_file)
    print 'Starting'
    execfile(config_file)
    main()
