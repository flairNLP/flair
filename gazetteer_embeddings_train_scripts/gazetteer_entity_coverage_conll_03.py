import torch
from flair.datasets import CONLL_03
from flair.embeddings import GazetteerEmbeddings

tokens = 0
tokens_found = 0
corpus = CONLL_03()
label_dict = corpus.make_label_dictionary(label_type='ner')
token_dict = {}
token_dict_2 = {}
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "./gazetteers",
                                                               partial_matching=True,
                                                               full_matching=False,
                                                               use_all_gazetteers=True,
                                                               label_dict=label_dict)
for sentence in corpus.get_all_sentences():
    gazetteer_embedding.embed(sentence)
    for token in sentence:
        try:
            if token_dict_2[token.text]:
                pass
        except KeyError:
            token_dict_2[token.text] = True
            tokens += 1
        try:
            if token.embedding[0] != torch.tensor(1):
                if token_dict[token.text][0]:
                    pass
        except IndexError:
            pass
        except KeyError:
            token_dict[token.text] = [True, token]
            tokens_found += 1

print(tokens)
print(tokens_found)

del gazetteer_embedding
del corpus
del label_dict
corpus = CONLL_03()
label_dict = corpus.make_label_dictionary(label_type='ner')
tokens = 0
tokens_found = 0

token_dict_3 = {}
token_dict_4 = {}

gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "./gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               use_all_gazetteers=True,
                                                               label_dict=label_dict)
for sentence in corpus.get_all_sentences():
    gazetteer_embedding.embed(sentence)
    for token in sentence:
        try:
            if token_dict_4[token.text]:
                pass
        except KeyError:
            token_dict_4[token.text] = True
            tokens += 1
        try:
            if token.embedding[0] != torch.tensor(1):
                if token_dict_3[token.text][0]:
                    pass
        except IndexError:
            pass
        except KeyError:
            token_dict_3[token.text] = [True, token]
            tokens_found += 1

print(tokens)
print(tokens_found)

all(map(token_dict_3.pop, token_dict))
print(token_dict_3)
