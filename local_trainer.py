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
    ], '_EN_BIOFLAIR'),
    ([
        WordEmbeddings('glove'),
        FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-forward/best-lm.pt'),
        FlairEmbeddings('/glusterfs/dfs-gfs-dist/weberple-pub/pm_pmc-backward/best-lm.pt'),
    ], '_GLOVE_BIOFLAIR'),
    ([
        WordEmbeddings('en'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ], '_EN_NEWSFLAIR'),
    ([
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ], '_GLOVE_NEWSFLAIR')
]

train_with_dev_list = [(True, '_trained_with_dev'), (False, '_trained_without_dev')]
mini_batch_size_list = [(32, '_32_batch')]
for mini_batch_size, batch_suffix in mini_batch_size_list:
    for train_with_dev, train_with_dev_suffix in train_with_dev_list:
        for embedding_type, embedding_suffix in embedding_types:
            embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_type)

            # 5. initialize sequence tagger
            from flair.models import SequenceTagger

            tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                    embeddings=embeddings,
                                                    tag_dictionary=tag_dictionary,
                                                    tag_type=tag_type,
                                                    use_crf=True)

            # 6. initialize trainer
            from flair.trainers import ModelTrainer

            trainer: ModelTrainer = ModelTrainer(tagger, corpus)

            # 7. start training
            trainer.train(
                'resources/taggers/negation-speculation-tagger{}{}{}'.format(embedding_suffix, train_with_dev_suffix,
                                                                             batch_suffix),
                learning_rate=0.1,
                mini_batch_size=mini_batch_size,
                max_epochs=150,
                initial_extra_patience=2,
                train_with_dev=train_with_dev,
                embeddings_storage_mode='gpu')