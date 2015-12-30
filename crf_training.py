from pprint import pprint
import os
import argparse

from model_config import *
from model_use import *
from crf_defs import *

config_file = 'example_config.py'
config = None


def main():
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


    #~ # train
    #~ accuracies_crf, preds_crf = train_model(train_data, dev_data, crf, config, params_crf, 'CRF')
    #~ # print results: accuracies
    #~ print '##### Parameters'
    #~ pprint(config.to_string().splitlines())
    #~ print '##### Train/dev accuracies: NN'
    #~ pprint(accuracies_nn)
    #~ print '##### Train/dev accuracies: CRF'
    #~ pprint(accuracies_crf)
    #~ # print results: F measures
    #~ for ep in range(config.num_predict, config.num_epochs + 1, config.num_predict):
        #~ print '---------- epoch', ep
        #~ # crf_predictions = [fuse_preds_crf(sent, pred, config)
                           #~ # for sent, pred in zip(dev_data, preds_crf[ep])]
        #~ sentences = preds_to_sentences(preds_crf[ep], config)
        #~ evaluate(sentences, 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing the models for \
                                     various parameter values')
    parser.add_argument("-conf", "--config_file",
                        help="location of configuration file")
    args = parser.parse_args()
    if args.config_file:
        config_file = os.path.abspath(args.config_file)
    execfile(config_file)
    print 'Starting'
    main()
