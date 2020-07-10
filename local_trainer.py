from flair.data import Corpus
from flair.datasets import BIOSCOPE
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

# 1. get the corpus
corpus: Corpus = BIOSCOPE()
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'tag'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [([
    WordEmbeddings('en'),
    FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-forward/best-lm.pt'),
    FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-backward/best-lm.pt'),
    ], 'EN_BIOFLAIR'),
    ([
        WordEmbeddings('glove'),
        FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-forward/best-lm.pt'),
        FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-backward/best-lm.pt'),
    ], 'GLOVE_BIOFLAIR'),
    ([
        WordEmbeddings('en'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ], 'EN_NEWSFLAIR'),
    ([
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ], 'GLOVE_NEWSFLAIR')
]

train_with_dev_list = [(True, '_trained_with_dev'), (False, '_trained_without_dev')]
mini_batch_size_list = [(16, '_16_batch'), (32, '_32_batch')]
for mini_batch_size, batch_suffix in mini_batch_size_list:
    for train_with_dev, train_with_dev_suffix in train_with_dev_list:
        for embedding_type, embedding_suffix in embedding_types:
            print('resources/taggers/negation-speculation-tagger{}{}{}'.format(embedding_suffix, train_with_dev_suffix, batch_suffix))