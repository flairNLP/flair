from flair.data import Sentence
from flair.models.entity_mention_linking import (
    Ab3PEntityPreprocessor,
    BioSynEntityPreprocessor,
    EntityMentionLinker,
    load_dictionary,
)
from flair.nn import Classifier


def test_bel_dictionary():
    """Check data in dictionary is what we expect.

    Hard to define a good test as dictionaries are DYNAMIC,
    i.e. they can change over time.
    """
    dictionary = load_dictionary("diseases")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.startswith(("MESH:", "OMIM:", "DO:DOID"))

    dictionary = load_dictionary("ctd-diseases")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.startswith("MESH:")

    dictionary = load_dictionary("ctd-chemicals")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.startswith("MESH:")

    dictionary = load_dictionary("chemical")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.startswith("MESH:")

    dictionary = load_dictionary("ncbi-taxonomy")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.isdigit()

    dictionary = load_dictionary("species")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.isdigit()

    dictionary = load_dictionary("ncbi-gene")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.isdigit()

    dictionary = load_dictionary("genes")
    candidate = dictionary.candidates[0]
    assert candidate.concept_id.isdigit()


def test_biosyn_preprocessing():
    """Check preprocessing does not produce empty strings."""
    preprocessor = BioSynEntityPreprocessor()

    # NOTE: Avoid emtpy string if mentions are just punctutations (e.g. `-` or `(`)
    for s in ["-", "(", ")", "9"]:
        assert len(preprocessor.process_mention(s)) > 0
        assert len(preprocessor.process_entity_name(s)) > 0


def test_abbrevitation_resolution():
    """Test abbreviation resolution works correctly."""
    preprocessor = Ab3PEntityPreprocessor(preprocessor=BioSynEntityPreprocessor())

    sentences = [
        Sentence("Features of ARCL type II overlap with those of Wrinkly skin syndrome (WSS)."),
        Sentence("Weaver-Smith syndrome (WSS) is a Mendelian disorder of the epigenetic machinery."),
    ]

    preprocessor.initialize(sentences)

    mentions = ["WSS", "WSS"]
    for idx, (mention, sentence) in enumerate(zip(mentions, sentences)):
        mention = preprocessor.process_mention(mention, sentence)
        if idx == 0:
            assert mention == "wrinkly skin syndrome"
        elif idx == 1:
            assert mention == "weaver smith syndrome"


def test_biomedical_entity_linking():
    sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

    tagger = Classifier.load("hunflair")
    tagger.predict(sentence)

    disease_linker = EntityMentionLinker.load("masaenger/bio-nen-disease")
    disease_dictionary = disease_linker.dictionary
    disease_linker.predict(sentence, pred_label_type="disease-nen", entity_label_types="diseases")

    gene_linker = EntityMentionLinker.load("masaenger/bio-nen-gene")
    gene_dictionary = gene_linker.dictionary
    gene_linker.predict(sentence,  pred_label_type="gene-nen", entity_label_types="genes")

    print("Diseases")
    for label in sentence.get_labels("disease-nen"):
        candidate = disease_dictionary[label.value]
        print(f"Candidate: {candidate.concept_name}")

    print("Genes")
    for label in sentence.get_labels("gene-nen"):
        candidate = gene_dictionary[label.value]
        print(f"Candidate: {candidate.concept_name}")
