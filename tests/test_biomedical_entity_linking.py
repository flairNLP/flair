import pytest

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
    dictionary = load_dictionary("disease")
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

    dictionary = load_dictionary("gene")
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


@pytest.mark.integration()
def test_biomedical_entity_linking():
    sentence = Sentence(
        "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
        "a neurodegenerative disease, which is exacerbated by exposure to high "
        "levels of mercury in dolphin populations.",
    )

    tagger = Classifier.load("hunflair")
    tagger.predict(sentence)

    linker = EntityMentionLinker.load("disease-linker")
    linker.predict(sentence)

    for span in sentence.get_spans():
        print(span)


def test_legacy_sequence_tagger():
    sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

    legacy_tagger = Classifier.load("hunflair")
    legacy_tagger.predict(sentence)

    disease_linker = EntityMentionLinker.load("hunflair/biosyn-sapbert-ncbi-disease")
    disease_linker.predict(sentence, pred_label_type="disease-nen")

    assert disease_linker._warned_legacy_sequence_tagger


if __name__ == "__main__":
    test_bel_dictionary()
