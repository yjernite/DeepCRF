# All the model arguments / parameters / file locations in one file
from os.path import join as pjoin
from utils import *


def base_config(input_features, l1_list, tag_list):
    return Config(input_features=input_features, l1_list=l1_list,
                  tag_list=tag_list)


def base_rnn_config(input_features, l1_list, tag_list):
    return Config(input_features=input_features, l1_list=l1_list,
                  tag_list=tag_list, use_rnn=True)


def base_convo_config(input_features, l1_list, tag_list):
    return Config(input_features=input_features, l1_list=l1_list,
                  tag_list=tag_list, use_convo=True,
                  num_epochs=15, num_predict=5, pred_window=3)


def base_crf_config(input_features, l1_list, tag_list):
    config = Config(input_features=input_features, l1_list=l1_list,
                    tag_list=tag_list, use_convo=True,
                    num_epochs=6, num_predict=2,
                    pred_window=3,
                    pot_window=3)
    config.features_dim = config.n_tags ** config.pot_window * config.pot_window
    return config


# file locations
git_dir = '/home/jernite/Code/DeepCRF'

train_file = pjoin(git_dir, 'Data/semeval_train/crfpp_text_batch_1.txt')
dev_file = pjoin(git_dir, 'Data/semeval_dev/crfpp_text_batch_1.txt')
vecs_file = pjoin(git_dir, 'Data/semeval_vecs.dat')

train_spans_file = pjoin(git_dir, 'Data/semeval_train/crfpp_spans_batch_1.txt')
dev_spans_file = pjoin(git_dir, 'Data/semeval_dev/crfpp_spans_batch_1.txt')

# feature names and tag list
features = ['word', 'lemma', 'pos', 'normal', 'word_length',
            'prefix', 'suffix', 'all_caps', 'capitalized', 'word_pos',
            'sentence_pos', 'sentence_length', 'med_prefix',
            'umls_match_tag_full', 'umls_match_tag_prefix',
            'umls_match_tag_acro', 'label']

input_features = ['lemma', 'prefix', 'suffix', 'pos', 'umls_match_tag_full']
l1_list = ['lemma', 'prefix', 'suffix']
tag_list = ['<P>', 'B', 'I', 'O', 'ID', 'OD']
