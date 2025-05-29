from flair.datasets import IMDB, AGNEWS

corpus_ag = AGNEWS()
corpus_ag.make_label_dictionary(label_type='topic', min_count=0, add_unk=False, add_dev_test=True)

# corpus_imdb = IMDB(rebalance_corpus=False, noise=False)