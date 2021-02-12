from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus: Corpus = CONLL_03()
tag_type = "ner"
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding = WordEmbeddings('glove')

tagger: SequenceTagger = SequenceTagger(hidden_size=256, # Model Parameter 1
                                        embeddings=embedding, # Model Parameter 2
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-pos',
              learning_rate=0.01, # Training Parameter 1
              mini_batch_size=32, # Training Parameter 2
              max_epochs=150)