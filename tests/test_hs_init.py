import os
import flair
import flair.datasets
from flair.embeddings import PooledFlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

flair.cache_root = os.path.expanduser(os.path.join("~", ".dtlearn"))

import torch

corpus = flair.datasets.CONLL_03()
print(corpus)

# Make classification tag dictionary for network output dimensions
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings and stack them
embedding_types = [
    # contextual string embeddings, forward
    PooledFlairEmbeddings('news-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('news-backward', pooling='min'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)


# Initialize NER model
tagger = SequenceTagger(
    hidden_size=256,embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
)

trainer = ModelTrainer(tagger, corpus)

# Start training
trainer.train(
    '../resources/conll03',
    max_epochs=10
)
