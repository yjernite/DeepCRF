input_features = ['lemma', 'prefix', 'suffix', 'pos', 'umls_match_tag_full']
l1_list = ['lemma', 'prefix', 'suffix']
tag_list = ['<P>', 'B', 'I', 'O', 'ID', 'OD']

config = base_crf_config(input_features, l1_list, tag_list)

config.learning_rate = 1e-3
config.l2_list = config.input_features

config.gradient_clip = 1
config.param_clip = 50

config.num_epochs = 12
