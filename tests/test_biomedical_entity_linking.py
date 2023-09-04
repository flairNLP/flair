from flair.data import Sentence
from flair.models.biomedical_entity_linking import (
    BiomedicalEntityLinkingDictionary,
    EntityMentionLinker,
)
from flair.nn import Classifier


def test_bel_dictionary():
    """Check data in dictionary is what we expect.

    Hard to define a good test as dictionaries are DYNAMIC,
    i.e. they can change over time.
    """
    dictionary = BiomedicalEntityLinkingDictionary.load("diseases")
    _, identifier = next(dictionary.stream())
    assert identifier.startswith(("MESH:", "OMIM:", "DO:DOID"))

    dictionary = BiomedicalEntityLinkingDictionary.load("ctd-diseases")
    _, identifier = next(dictionary.stream())
    assert identifier.startswith("MESH:")

    dictionary = BiomedicalEntityLinkingDictionary.load("ctd-chemicals")
    _, identifier = next(dictionary.stream())
    assert identifier.startswith("MESH:")

    dictionary = BiomedicalEntityLinkingDictionary.load("chemical")
    _, identifier = next(dictionary.stream())
    assert identifier.startswith("MESH:")

    dictionary = BiomedicalEntityLinkingDictionary.load("ncbi-taxonomy")
    _, identifier = next(dictionary.stream())
    assert identifier.isdigit()

    dictionary = BiomedicalEntityLinkingDictionary.load("species")
    _, identifier = next(dictionary.stream())
    assert identifier.isdigit()

    dictionary = BiomedicalEntityLinkingDictionary.load("ncbi-gene")
    _, identifier = next(dictionary.stream())
    assert identifier.isdigit()

    dictionary = BiomedicalEntityLinkingDictionary.load("genes")
    _, identifier = next(dictionary.stream())
    assert identifier.isdigit()


def test_biomedical_entity_linking():
    sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

    tagger = Classifier.load("hunflair")
    tagger.predict(sentence)

    disease_linker = EntityMentionLinker.load("diseases", "diseases-nel", hybrid_search=True)
    disease_dictionary = disease_linker.candidate_generator.dictionary
    disease_linker.predict(sentence)

    gene_linker = EntityMentionLinker.load("genes", "genes-nel", hybrid_search=False, entity_type="genes")
    gene_dictionary = gene_linker.candidate_generator.dictionary

    gene_linker.predict(sentence)

    print("Diseases")
    for span in sentence.get_spans(disease_linker.entity_type):
        print(f"Span: {span.text}")
        for candidate_label in span.get_labels(disease_linker.label_type):
            candidate = disease_dictionary[candidate_label.value]
            print(f"Candidate: {candidate.concept_name}")

    print("Genes")
    for span in sentence.get_spans(gene_linker.entity_type):
        print(f"Span: {span.text}")
        for candidate_label in span.get_labels(gene_linker.label_type):
            candidate = gene_dictionary[candidate_label.value]
            print(f"Candidate: {candidate.concept_name}")

    breakpoint()  # noqa: T100
