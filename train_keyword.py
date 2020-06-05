from typing import List

import flair.datasets
from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    BertEmbeddings,
    TransformerWordEmbeddings
)
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

data = [
        (flair.datasets.INSPEC(), 'keyphrase-tagger-inspec'),
        (flair.datasets.SEMEVAL2010(), 'keyphrase-tagger-semeval2010'),
        (flair.datasets.SEMEVAL2017(), 'keyphrase-tagger-semeval2017')
        ]

for corpus_object, path in data:
    # 1. get the corpus
    corpus: Corpus = corpus_object
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = "keyword"

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)


    # initialize embeddings
    embedding_types = \
    [
        [
            WordEmbeddings("crawl"),
            FlairEmbeddings("news-forward"),
            FlairEmbeddings("news-backward")
        ],
        [
            TransformerWordEmbeddings('/tmp/scibert_scivocab_uncased')
        ]
    ]

    for embedding_type in embedding_types:

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_type)

        # initialize sequence tagger
        from flair.models import SequenceTagger

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=True,
        )

        # initialize trainer
        from flair.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(
            "resources/taggers/{}".format(path),
            learning_rate=0.05,
            mini_batch_size=16,
            max_epochs=150,
            embeddings_storage_mode='gpu'
        )
        """x
        plotter = Plotter()
        plotter.plot_training_curves("resources/taggers/{}/loss.tsv".format(path))
        plotter.plot_weights("resources/taggers/{}/weights.txt".format(path))
        """
