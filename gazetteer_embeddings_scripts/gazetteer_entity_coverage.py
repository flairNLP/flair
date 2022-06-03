import torch
from flair.datasets import CONLL_03, WNUT_17, NER_ENGLISH_STACKOVERFLOW
from flair.embeddings import GazetteerEmbeddings
from contextlib import redirect_stdout

with open('out.txt', 'w') as f:
    with redirect_stdout(f):
        path = "gazetteers/"
        tokenize = True
        use_all_gazetteers = True

        settings_dict = [[True, True, 'train'], [True, False, 'train'], [False, True, 'train'],
                        [True, True, 'test'], [True, False, 'test'], [False, True, 'test']]
        print("CONLL_03")
        for setting in settings_dict:
            tokens = 0
            tokens_found = 0
            corpus = CONLL_03()
            label_dict = corpus.make_label_dictionary(label_type='ner')
            token_set_1 = set()
            token_set_2 = set()
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=path,
                                                                        partial_matching=setting[0],
                                                                        full_matching=setting[1],
                                                                        tokenize_gazetteer_entries=tokenize,
                                                                        use_all_gazetteers=use_all_gazetteers,
                                                                        label_dict=label_dict)
            for sentence in getattr(corpus, setting[2]):
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
            print(f"Partial: {setting[0]} || Full: {setting[1]}")
            print(len(token_set_1))
            print(len(token_set_2))
            print(f"{setting[2]}: {((len(token_set_2)/len(token_set_1))*100)}")

            del gazetteer_embedding
            del corpus
            del label_dict
            del token_set_1
            del token_set_2

        print("WNUT_17")
        for setting in settings_dict:
            tokens = 0
            tokens_found = 0
            corpus = WNUT_17()
            label_dict = corpus.make_label_dictionary(label_type='ner')
            token_set_1 = set()
            token_set_2 = set()
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=path,
                                                                        partial_matching=setting[0],
                                                                        full_matching=setting[1],
                                                                        tokenize_gazetteer_entries=tokenize,
                                                                        use_all_gazetteers=use_all_gazetteers,
                                                                        label_dict=label_dict)
            for sentence in getattr(corpus, setting[2]):
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
            print(f"Partial: {setting[0]} || Full: {setting[1]}")
            print(len(token_set_1))
            print(len(token_set_2))
            print(f"{setting[2]}: {((len(token_set_2)/len(token_set_1))*100)}")

            del gazetteer_embedding
            del corpus
            del label_dict
            del token_set_1
            del token_set_2