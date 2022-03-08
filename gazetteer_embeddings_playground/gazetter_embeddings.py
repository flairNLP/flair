import flair.datasets
from flair.datasets import WNUT_17, CONLL_03
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# corpus1 = WNUT_17()
# corpus = CONLL_03()
sentences_1 = Sentence('I love Sandys Fort Spring!')
sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
sentence_list = [sentences_1, sentences_2]

label_list = ['PER', 'ORG', 'LOC', 'MISC']
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "./gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               label_list=label_list)
# gazetteer_embedding.embed(sentence_list)
print(gazetteer_embedding.feature_list)

stacked_embeddings = StackedEmbeddings([bert_embedding, gazetteer_embedding])

stacked_embeddings.embed(sentence_list)

for sentence in sentence_list:
    for token in sentence:
        print(token)
        print(token.embedding)
