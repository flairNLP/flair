import torch
from flair.datasets import CONLL_03, WNUT_17, NER_ENGLISH_STACKOVERFLOW
from flair.embeddings import GazetteerEmbeddings

tokens = 0
tokens_found = 0
corpus = CONLL_03()
label_dict = corpus.make_label_dictionary(label_type='ner')
token_set_1 = set()
token_set_2 = set()
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                               "./count_sensitive_gazetteers",
                                                               partial_matching=True,
                                                               full_matching=True,
                                                               use_all_gazetteers=False,
                                                               label_dict=label_dict)
for sentence in corpus.get_all_sentences():
    gazetteer_embedding.embed(sentence)
    for token in sentence:
        if token.text in token_set_1:
            pass
        else:
            token_set_1.add(token.text)
        try:
            if token.embedding[0] != torch.tensor(1) and 1 in token.embedding[1:]:
                if token.text in token_set_2:
                    pass
                else:
                    token_set_2.add(token.text)
        except IndexError:
            pass

print(len(token_set_1))
print(len(token_set_2))
print(((len(token_set_2)/len(token_set_1))*100))
