from flair.datasets import CONLL_03
from flair.embeddings import StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence

corpus = CONLL_03()
sentences_1 = Sentence('Jack in the Box is an American fast-food restaurant chain founded in 1951.')
sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
sentence_list = [sentences_1, sentences_2]

label_dict = corpus.make_label_dictionary(label_type='ner')
# Example use of the GazetteerEmbeddings Class
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               label_dict=None,
                                                               use_all_gazetteers=True,
                                                               tokenize_gazetteer_entries=False)


# label_dict only needed if selected set of gazetteers is used -> that would be the case if use_all_gazetteers=False
# if use_all_gazetteers=True, -> label_dict can be left to None, every gazetteer will be left as a single source of
# information, all gazetteers will be included, regardless of their filename
# if use_all_gazetteers=False -> the gazetteer files need to have the entity type in the name and
# need to be a .txt file, such as "LOC_entities.txt".
# if tokenize_gazetteer_entries=True -> the entities of a gazetteer will be tokenized and then loaded into the dict
# if tokenize_gazetteer_entries=False -> the entities of a gazetteer will be simply split by white space

print(gazetteer_embedding.feature_list)

stacked_embeddings = StackedEmbeddings([gazetteer_embedding])

stacked_embeddings.embed(sentence_list)

for sentence in sentence_list:
    for token in sentence:
        print(token)
        print(token.embedding)
