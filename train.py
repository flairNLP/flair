from typing import List

import flair.datasets
from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
)
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

# 1. get the corpus
corpus: Corpus = flair.datasets.UD_ENGLISH()
print(corpus)

# 2. what tag do we want to predict?
tag_type = "upos"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings("glove"),
    # comment in this line to use character embeddings
    # CharacterEmbeddings(),
    # comment in these lines to use contextual string embeddings
    #
    # FlairEmbeddings('news-forward'),
    #
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

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
    "resources/taggers/example-ner",
    EvaluationMetric.MICRO_F1_SCORE,
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=20,
    shuffle=False,
)

plotter = Plotter()
plotter.plot_training_curves("resources/taggers/example-ner/loss.tsv")
plotter.plot_weights("resources/taggers/example-ner/weights.txt")
