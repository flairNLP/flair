# from flair.data import Sentence
# from flair.models.biomedical_entity_linking import BioNelDictionary
# from flair.nn import Classifier
# from flair.tokenization import SciSpacyTokenizer

# def test_bionel_dictionary():
#     """
#     Check data in dictionary is what we expect.
#     Hard to define a good test as dictionaries are DYNAMIC,
#     i.e. they can change over time
#     """

#     dictionary = BioNelDictionary.load("disease")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith(("MESH:", "OMIM:", "DO:DOID"))

#     dictionary = BioNelDictionary.load("ctd-disease")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BioNelDictionary.load("ctd-chemical")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BioNelDictionary.load("chemical")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BioNelDictionary.load("ncbi-taxonomy")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BioNelDictionary.load("species")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BioNelDictionary.load("ncbi-gene")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BioNelDictionary.load("gene")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()


# def test_biomedical_entity_linking():
#     sentence = Sentence(
#         "Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer()
#     )
#     ner_tagger = Classifier.load("hunflair-disease")
#     ner_tagger.predict(sentence)
#     nen_tagger = BiomedicalEntityLinker.load("disease")
#     nen_tagger.predict(sentence)
#     for tag in sentence.get_labels():
#         print(tag)


# if __name__ == "__main__":
#     test_bionel_dictionary()
# test_biomedical_entity_linking()
