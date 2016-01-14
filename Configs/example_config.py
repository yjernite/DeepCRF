from os.path import join as pjoin

# file locations
git_dir = '/home/jernite/Code/DeepCRF'

data_dir = pjoin(git_dir, 'Data/crf_data_overlaps/')

train_file = pjoin(data_dir, 'semeval_train/crfpp_text_batch_1.txt')
dev_file = pjoin(data_dir, 'semeval_dev/crfpp_text_batch_1.txt')
vecs_file = pjoin(git_dir, 'Data/crf_data_overlaps/semeval_vecs.dat')

# feature names and tag list
features = ['word', 'lemma', 'pos', 'normal', 'word_length',
            'prefix', 'suffix', 'all_caps', 'capitalized', 'word_pos',
            'sentence_pos', 'sentence_length', 'med_prefix',
            'umls_match_tag_full', 'umls_match_tag_prefix',
            'umls_match_tag_acro', 'label']

input_features = ['lemma', 'prefix', 'suffix', 'pos', 'umls_match_tag_full']
l1_list = ['lemma', 'prefix', 'suffix']
tag_list = ['<P>', 'B', 'I', 'O', 'ID', 'OD']

input_features = ['word', 'lemma', 'pos', 'normal', 'word_length',
                  'prefix', 'suffix', 'capitalized', 'word_pos',
                  'sentence_pos', 'sentence_length', 'med_prefix',
                  'umls_match_tag_full', 'umls_match_tag_prefix',
                  'umls_match_tag_acro']

l1_list = ['word', 'lemma', 'normal', 'prefix', 'suffix']
tag_list = ['<P>', 'B', 'Bp', 'I', 'Ip', 'In', 'ID', 'O', 'OD']

config = Config()

config.learning_rate = 1e-3
config.l2_list = config.input_features

config.gradient_clip = 1
config.param_clip = 50

config.num_epochs = 12
