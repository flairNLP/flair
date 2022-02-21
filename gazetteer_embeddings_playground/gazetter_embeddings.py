import flair.datasets
from flair.datasets import WNUT_17, CONLL_03
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from datasets import list_datasets, load_dataset, list_metrics, load_metric

# corpus1 = WNUT_17()
corpus2 = CONLL_03()
# print(corpus)
#
# label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
# dataset = load_dataset("conll2003")
# print(dataset['train'][0])
#

sentence = Sentence('The grass is green .')
sentence1 = Sentence('I love Paris !')
sentence_list = [sentence, sentence1]
# glove_embedding = WordEmbeddings('glove')
#
# glove_embedding.embed(sentence)
#
# # now check out the embedded tokens.
# for token in sentence:
#     print(token)
#     print(token.embedding)
#
# transformer_embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
#                                                    layers="-1",
#                                                    subtoken_pooling="first",
#                                                    fine_tune=True,
#                                                    use_context=True,
#                                                    )
#
# transformer_embeddings.embed(sentence)
#
# for token in sentence:
#     print(token)
#     print(token.embedding)

##########The Goal############

# # somehow load a gazetteer
# loaded_gazetteer = load(...)
#
# init embedding with gazetteer
# gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/home/danielc/PycharmProjects"
#                                                                                   "/flair"
#                                                                                   "/gazetteer_embeddings_playground"
#                                                                                   "/gazetteers",
#                                                                label_dict=label_dict)
# print(gazetteer_embedding.gazetteer_file_list)
#
# print(gazetteer_embedding.embedding_length)
#
# gazetteer_embedding.embed(sentence_list)
#
# for sentence in sentence_list:
#     for token in sentence:
#         print(token)
#         print(token.embedding)

# embeddings_list = [gazetteer_embedding, transformer_embeddings]
#
# stacked_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings_list)
#
# sentence = Sentence('The grass is not green .')
#
# stacked_embeddings.embed(sentence)
#
# for token in sentence:
#     print(token)
#     print(token.embedding)
