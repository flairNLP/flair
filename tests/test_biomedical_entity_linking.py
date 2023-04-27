# from flair.data import Sentence
# from flair.models.biomedical_entity_linking import (
#     BiomedicalEntityLinker,
#     BiomedicalEntityLinkingDictionary,
# )
# from flair.nn import Classifier


# def test_bel_dictionary():
#     """
#     Check data in dictionary is what we expect.
#     Hard to define a good test as dictionaries are DYNAMIC,
#     i.e. they can change over time
#     """

#     dictionary = BiomedicalEntityLinkingDictionary.load("disease")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith(("MESH:", "OMIM:", "DO:DOID"))

#     dictionary = BiomedicalEntityLinkingDictionary.load("ctd-disease")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BiomedicalEntityLinkingDictionary.load("ctd-chemical")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BiomedicalEntityLinkingDictionary.load("chemical")
#     _, identifier = next(dictionary.stream())
#     assert identifier.startswith("MESH:")

#     dictionary = BiomedicalEntityLinkingDictionary.load("ncbi-taxonomy")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BiomedicalEntityLinkingDictionary.load("species")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BiomedicalEntityLinkingDictionary.load("ncbi-gene")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()

#     dictionary = BiomedicalEntityLinkingDictionary.load("gene")
#     _, identifier = next(dictionary.stream())
#     assert identifier.isdigit()


# def test_biomedical_entity_linking():

#     sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

#     tagger = Classifier.load("hunflair")
#     tagger.predict(sentence)

#     disease_linker = BiomedicalEntityLinker.load("disease", hybrid_search=True)
#     disease_linker.predict(sentence)

#     gene_linker = BiomedicalEntityLinker.load("gene", hybrid_search=False)

#     breakpoint()


# if __name__ == "__main__":
#     # test_bel_dictionary()
#     test_biomedical_entity_linking()
