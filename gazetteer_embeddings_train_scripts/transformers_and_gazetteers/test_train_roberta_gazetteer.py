from flair.embeddings import TransformerWordEmbeddings, GazetteerEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import CONLL_03

corpus = CONLL_03()

label_type = 'ner'

label_dict = corpus.make_label_dictionary(label_type=label_type)

print(label_dict)

label_list = ['PER', 'ORG', 'LOC', 'MISC']

roberta_embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                               layers="-1",
                                               subtoken_pooling="first",
                                               fine_tune=True,
                                               use_context=True)

gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "../gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               label_list=label_list)

stacked_embeddings = StackedEmbeddings([roberta_embeddings, gazetteer_embedding])

tagger = SequenceTagger(hidden_size=256,
                        embeddings=stacked_embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=True,
                        use_rnn=True,
                        reproject_embeddings=True)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/gazetteer_roberta',
                  learning_rate=5.0e-6,
                  mini_batch_size=4)
