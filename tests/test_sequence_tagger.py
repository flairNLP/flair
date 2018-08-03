from flair.data import Sentence
from flair.models import SequenceTagger


def test_tag_sentence():

    # test tagging
    sentence = Sentence('I love Berlin')

    tagger = SequenceTagger.load('ner')

    tagger.predict(sentence)

    # test re-tagging
    tagger = SequenceTagger.load('pos')

    tagger.predict(sentence)
