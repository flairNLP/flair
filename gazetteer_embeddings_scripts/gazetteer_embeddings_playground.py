import flair.datasets
from flair.datasets import WNUT_17, CONLL_03, NER_ENGLISH_STACKOVERFLOW
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# corpus = WNUT_17()
corpus = CONLL_03()
# corpus = NER_ENGLISH_STACKOVERFLOW()

sentences_1 = Sentence('Jack in the Box is an American fast-food restaurant chain founded in 1951.')
sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
sentence_list = [sentences_1, sentences_2]

label_dict = corpus.make_label_dictionary(label_type='ner')

gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "./gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               label_dict=label_dict,
                                                               use_all_gazetteers=False)
# gazetteer_embedding.embed(sentence_list)
print(gazetteer_embedding.feature_list)

stacked_embeddings = StackedEmbeddings([gazetteer_embedding])

stacked_embeddings.embed(sentence_list)

for sentence in sentence_list:
    for token in sentence:
        print(token)
        print(token.embedding)
