from flair.data import Sentence


def test_sentence_context():
    # make a sentence and some right context
    sentence = Sentence("George Washington ging nach Washington.")
    sentence._next_sentence = Sentence("Das ist eine schöne Stadt.")

    assert sentence.right_context(1) == ["Das"]
    assert sentence.right_context(10) == ["Das", "ist", "eine", "schöne", "Stadt", "."]


def test_equality():
    assert Sentence("Guten Tag!") != Sentence("Good day!")
    assert Sentence("Guten Tag!", use_tokenizer=True) != Sentence("Guten Tag!", use_tokenizer=False)

    # TODO: is this desirable? Or should two sentences with same text still be considered different objects?
    assert Sentence("Guten Tag!") == Sentence("Guten Tag!")


def test_to_dict():
    s = Sentence("This is a very interesting sentence is it not")
    s[1].add_label("ner", "B-Foo")
    s[2].add_label("ner", "B-Bar")
    s[3].add_label("ner", "I-Bar")
    s[4].add_label("ner", "I-Bar")
    s[5].add_label("ner", "E-Bar")

    assert s.to_dict() == {
        "text": "This is a very interesting sentence is it not",
        "all labels": [
            {"value": "B-Foo", "confidence": 1.0},
            {"value": "B-Bar", "confidence": 1.0},
            {"value": "I-Bar", "confidence": 1.0},
            {"value": "I-Bar", "confidence": 1.0},
            {"value": "E-Bar", "confidence": 1.0},
        ],
    }
    assert s.to_dict("ner") == {
        "text": "This is a very interesting sentence is it not",
        "ner": [
            {"value": "B-Foo", "confidence": 1.0},
            {"value": "B-Bar", "confidence": 1.0},
            {"value": "I-Bar", "confidence": 1.0},
            {"value": "I-Bar", "confidence": 1.0},
            {"value": "E-Bar", "confidence": 1.0},
        ],
    }
    assert s.to_dict("ner", use_spans=True) == {
        "text": "This is a very interesting sentence is it not",
        "ner": [
            {"value": "Foo", "start_pos": 5, "end_pos": 7, "confidence": 1.0},
            {"value": "Bar", "start_pos": 8, "end_pos": 35, "confidence": 1.0},
        ],
    }
