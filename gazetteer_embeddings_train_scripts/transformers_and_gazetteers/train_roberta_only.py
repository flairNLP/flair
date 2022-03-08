from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import CONLL_03

corpus = CONLL_03()

label_type = 'ner'

label_dict = corpus.make_label_dictionary(label_type=label_type)

print(label_dict)

roberta_embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                               layers="-1",
                                               subtoken_pooling="first",
                                               fine_tune=True,
                                               use_context=True)

tagger = SequenceTagger(hidden_size=256,
                        embeddings=roberta_embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=True,
                        use_rnn=True,
                        reproject_embeddings=True)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/only_roberta',
                  learning_rate=5.0e-6,
                  mini_batch_size=4)
