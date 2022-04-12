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
