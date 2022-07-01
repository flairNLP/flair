from flair.datasets import CONLL_03
from flair.embeddings import StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence

corpus = CONLL_03()
sentences_1 = Sentence('Jack in the Box is an American fast-food restaurant chain founded in 1951.')
sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
sentence_list = [sentences_1, sentences_2]

label_dict = corpus.make_label_dictionary(label_type='ner')
# Example use of the GazetteerEmbeddings Class
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers= # the path to the gazetteer directory
                                                               "gazetteer-collection",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               label_dict=None, # label_dict only needed if selected set of gazetteers is used -> that would be the case if use_all_gazetteers=False
                                                               use_all_gazetteers=True) # use all gazetteers of a dir, -> label_dict can be left to None
print(gazetteer_embedding.feature_list)

stacked_embeddings = StackedEmbeddings([gazetteer_embedding])

stacked_embeddings.embed(sentence_list)

for sentence in sentence_list:
    for token in sentence:
        print(token)
        print(token.embedding)
