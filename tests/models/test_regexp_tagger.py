from flair.data import Sentence
from flair.models import RegexpTagger


def test_regexp_tagger():

    sentence = Sentence('Der sagte: "das ist durchaus interessant"')

    tagger = RegexpTagger(
        mapping=[(r'["„»]((?:(?=(\\?))\2.)*?)[”"“«]', "quote_part", 1), (r'["„»]((?:(?=(\\?))\2.)*?)[”"“«]', "quote")]
    )

    tagger.predict(sentence)

    assert sentence.get_label("quote_part").data_point.text == "das ist durchaus interessant"
    assert sentence.get_label("quote").data_point.text == '"das ist durchaus interessant"'
