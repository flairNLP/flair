import os

import pytest

from flair.data import Sentence
from flair.models import SequenceTagger

@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_tag_sentence():

    # test tagging
    sentence = Sentence('I love Berlin')

    tagger = SequenceTagger.load('ner')

    tagger.predict(sentence)

    # test re-tagging
    tagger = SequenceTagger.load('pos')

    tagger.predict(sentence)
