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

tag_list = ['<P>', 'B', 'Bp', 'I', 'Ip', 'In', 'ID', 'O', 'OD']

input_features = ['word', 'lemma', 'pos', 'normal', 'word_length',
                  'prefix', 'suffix', 'capitalized', 'word_pos',
                  'sentence_pos', 'sentence_length', 'med_prefix',
                  'umls_match_tag_full', 'umls_match_tag_prefix',
                  'umls_match_tag_acro']


config = Config(input_features=input_features, tag_list=tag_list)

config.l1_list = ['word', 'lemma', 'normal', 'prefix', 'suffix']

config.learning_rate = 5e-3
config.l2_list = config.input_features

config.gradient_clip = 5
config.param_clip = 50

config.num_epochs = 12

config.optimizer = 'adam'
config.batch_size = 10
