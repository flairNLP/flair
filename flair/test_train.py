from flair.models import BiaffineTager
from flair.datasets import CONLL_03
import flair
import torch

flair.device = torch.device("cuda:2")

corpus = CONLL_03('../resources/')
tag_dictionary = corpus.make_label_dictionary('ner')

from flair.embeddings import TransformerWordEmbeddings

embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

from flair.trainers import ModelTrainer


tagger = BiaffineTager(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=False,
    use_rnn=True,
    use_biaffine=True,
    reproject_embeddings=False
)

from torch.optim.lr_scheduler import OneCycleLR

trainer = ModelTrainer(tagger, corpus)

trainer.train('/vol/fob-vol7/mi19/chenlei/nlp/biaffine',
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=2,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              )
