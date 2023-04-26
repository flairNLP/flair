from flair.data import Sentence
from flair.models.biomedical_entity_linking import BiomedicalEntityLinker
from flair.nn import Classifier
from flair.tokenization import SciSpacyTokenizer


def test_biomedical_entity_linking():
    sentence = Sentence(
        "Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer()
    )
    ner_tagger = Classifier.load("hunflair-disease")
    ner_tagger.predict(sentence)
    nen_tagger = BiomedicalEntityLinker.load("disease")
    nen_tagger.predict(sentence)
    for tag in sentence.get_labels():
        print(tag)


if __name__ == "__main__":
    test_biomedical_entity_linking()
