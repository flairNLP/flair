import torch.optim

import flair.datasets
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings

# 1. get the corpus
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import RelationClassifierLinear

corpus: Corpus = flair.datasets.SEMEVAL_2010_TASK_8(in_memory=False).downsample(0.1)
print(corpus)

# 3. make the tag dictionary from the corpus
relation_label_dict = corpus.make_relation_label_dictionary(label_type="label")
print(relation_label_dict.idx2item)

# initialize embeddings
embeddings = TransformerWordEmbeddings(layers="-1", fine_tune=False)

# initialize sequence tagger

model: RelationClassifierLinear = RelationClassifierLinear(
    # hidden_size=64,
    token_embeddings=embeddings,
    label_dictionary=relation_label_dict,
    label_type="label",
    span_label_type="ner",
)

# evaluate = model.evaluate(corpus.dev)
# print(evaluate)

# initialize trainer
from flair.trainers import ModelTrainer

# initialize trainer
trainer: ModelTrainer = ModelTrainer(model, corpus, optimizer=torch.optim.Adam)

trainer.train(
    "resources/classifiers/example-rc-backup",
    learning_rate=3e-5,
    mini_batch_size=4,
    mini_batch_chunk_size=1,
    max_epochs=10,
    shuffle=True,
)