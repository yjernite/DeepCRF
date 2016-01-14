from pprint import pprint
import os
import argparse

from model_use import *
from crf_defs import *

config_file = 'Configs/example_config.py'
config = None


def main():
    execfile(config_file)
    # load the data
    train_data = read_data(train_file, features, config)
    dev_data = read_data(dev_file, features, config)
    config.make_mappings(train_data + dev_data)
    # initialize the parameters
    if config.init_words:
        word_vectors = read_vectors(vecs_file, config.feature_maps['word']['reverse'])
        pre_trained = {'word': word_vectors}
    else:
        pre_trained = {}
    params_crf = Parameters(init=pre_trained)
    # make the CRF and SequNN models
    sess = tf.InteractiveSession()
    crf = CRF(config)
    crf.make(config, params_crf)
    sess.run(tf.initialize_all_variables())
    train_data_32 = cut_and_pad(train_data, config)
    for i in range(100):
        shuffle(train_data_32)
        print 'training', i
        crf.train_epoch(train_data_32, config, params_crf)
        print 'tagging', i
        preds = tag_dataset(dev_data, config, params_crf, 'CRF')
        sentences = preds_to_sentences(preds, config)
        print 'epoch', i
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
    main()
