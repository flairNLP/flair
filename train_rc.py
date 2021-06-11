from typing import List

import flair.datasets
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

# 1. get the corpus
corpus: Corpus = flair.datasets.SEMEVAL_2010_TASK_8()
print(corpus)

# 3. make the tag dictionary from the corpus
relation_label_dict = corpus.make_relation_label_dictionary(label_type="label")
print(relation_label_dict.idx2item)

# initialize embeddings
embeddings = TransformerWordEmbeddings()

# initialize sequence tagger
from flair.models import RelationClassifier

model: RelationClassifier = RelationClassifier(
    hidden_size=64,
    token_embeddings=embeddings,
    label_dictionary=relation_label_dict,
    label_type="label",
    span_label_type="ner",
)

# initialize trainer
from flair.trainers import ModelTrainer

# initialize trainer
trainer: ModelTrainer = ModelTrainer(model, corpus)

trainer.train(
    "resources/classifiers/example-rc",
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=10,
    # shuffle=False,
    shuffle=True,
)

plotter = Plotter()
plotter.plot_training_curves("resources/taggers/example-ner/loss.tsv")
plotter.plot_weights("resources/taggers/example-ner/weights.txt")
