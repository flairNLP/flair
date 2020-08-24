from flair.datasets import CONLL_03
from flair.data import Corpus
from flair.embeddings import POSEmbeddings, StackedEmbeddings, WordEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

embedding_types = [
    [POSEmbeddings(),],
    [WordEmbeddings('glove'),],
    [

    # GloVe embeddings
    POSEmbeddings(),

    # contextual string embeddings, forward
    FlairEmbeddings('news-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('news-backward'),
    ],
    [
    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    FlairEmbeddings('news-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('news-backward'),
    ]]


for idx, embedding_type in enumerate(embedding_types):

    corpus: Corpus = CONLL_03()

    tag_type = 'ner'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_type)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(f'resources/taggers/ner-tagger-with-pos-embeddings-{idx}',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150,
                  embeddings_storage_mode='gpu')