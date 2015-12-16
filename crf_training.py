from pprint import pprint
import os
import argparse

from model_config import *
from model_use import *
from crf_defs import *

config_file = 'example_config.py'
config = None

def combine_toks(tok_crf, tok_nn, mix):
    res = {}
    res['word'] = tok_nn['word']
    res['label'] = tok_nn['label']
    for tag in ['B', 'I', 'ID', 'O', 'OD']:
        res[tag] = mix * tok_crf[tag] + (1 - mix) * tok_nn[tag]
    return res


def main():
    # load the data
    train_data = read_data(train_file, features, config)
    dev_data = read_data(dev_file, features, config)
    dev_spans = treat_spans(dev_spans_file)
    config.make_mappings(train_data + dev_data)
    # initialize the parameters
    if config.init_words:
        word_vectors = read_vectors(vecs_file, config.feature_maps['word']['reverse'])
        pre_trained = {'word': word_vectors}
    else:
        pre_trained = {}
    params_crf = Parameters(init=pre_trained)
    params_nn = Parameters(init=pre_trained)
    # make the CRF and SequNN models
    sess = tf.InteractiveSession()
    crf = CRF(config)
    crf.make(config, params_crf)
    sequ_nn = SequNN(config)
    sequ_nn.make(config, params_nn)
    sess.run(tf.initialize_all_variables())
    # train
    accuracies_nn, preds_nn = train_model(train_data, dev_data, sequ_nn, config, params_nn, 'sequ_nn')
    accuracies_crf, preds_crf = train_model(train_data, dev_data, crf, config, params_crf, 'CRF')
    # print results: accuracies
    print '##### Parameters'
    pprint(config.to_string().splitlines())
    print '##### Train/dev accuracies: NN'
    pprint(accuracies_nn)
    print '##### Train/dev accuracies: CRF'
    pprint(accuracies_crf)
    # print results: F measures
    for ep in range(config.num_predict, config.num_epochs + 1, config.num_predict):
        print '---------- epoch', ep
        crf_predictions = [fuse_preds_crf(sent, pred, config)
                           for sent, pred in zip(dev_data, preds_crf[ep])]
        nn_predictions = [fuse_preds(sent, pred, config)
                          for sent, pred in zip(dev_data, preds_nn[ep])]
        merged_nn = merge(nn_predictions, dev_spans)
        merged_crf = merge(crf_predictions, dev_spans)
        predictions = [[combine_toks(tok_crf, tok_nn, 0.5)
                        for tok_crf, tok_nn in zip(sent_crf, sent_nn)]
                       for sent_crf, sent_nn in zip (crf_predictions, nn_predictions)]
        merged = merge(predictions, dev_spans)
        print '##### P-R-F curves: NN'
        for i in range(10):
            evaluate(merged_nn, 0.1 * i)
        print '##### P-R-F curves: CRF'
        for i in range(10):
            evaluate(merged_crf, 0.1 * i)
        print '##### P-R-F curves: AVG'
        for i in range(10):
            evaluate(merged, 0.1 * i)


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
