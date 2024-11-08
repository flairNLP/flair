from flair.data import Sentence


def test_sentence_context():
    # make a sentence and some right context
    sentence = Sentence("George Washington ging nach Washington.")
    sentence._next_sentence = Sentence("Das ist eine schöne Stadt.")

    assert sentence.right_context(1) == [sentence._next_sentence[0]]
    assert sentence.right_context(10) == sentence._next_sentence.tokens[:10]


def test_equality():
    assert Sentence("Guten Tag!") != Sentence("Good day!")
    assert Sentence("Guten Tag!", use_tokenizer=True) != Sentence("Guten Tag!", use_tokenizer=False)
    sentence1 = Sentence("This sentence will be labeled")
    sentence1[1].set_label("ner", "B-subject")
    sentence2 = Sentence("This sentence will be labeled")
    sentence2[1].set_label("ner", "B-object")
    assert sentence1 != sentence2

    assert Sentence("Guten Tag!") == Sentence("Guten Tag!")
    sentence2[1].set_label("ner", "B-subject")
    assert sentence1 == sentence2


def test_token_labeling():
    sentence = Sentence("This sentence will be labled")
    assert sentence.get_labels("ner") == []
    assert sentence.get_labels() == []
    sentence[2].add_label("ner", "B-promise")
    sentence[3].add_label("ner", "I-promise")
    sentence[4].add_label("ner", "I-promise")
    assert [label.value for label in sentence.get_labels()] == ["B-promise", "I-promise", "I-promise"]
    assert [token.get_label("ner").value for token in sentence] == ["O", "O", "B-promise", "I-promise", "I-promise"]
    sentence[1].set_label("ner", "B-object")
    sentence[1].set_label("ner", "B-subject")
    assert [label.value for label in sentence.get_labels("ner")] == ["B-subject", "B-promise", "I-promise", "I-promise"]
    assert [token.get_label("ner").value for token in sentence] == [
        "O",
        "B-subject",
        "B-promise",
        "I-promise",
        "I-promise",
    ]
    sentence.set_label("class", "positive")
    sentence.remove_labels("ner")
    assert sentence.get_labels("ner") == []
    assert [label.value for label in sentence.get_labels()] == ["positive"]
    sentence[0].add_label("pos", "first")
    sentence[0].add_label("pos", "primero")
    sentence[0].add_label("pos", "erstes")
    assert [label.value for label in sentence.get_labels("pos")] == ["first", "primero", "erstes"]
    assert sentence[0].get_label("pos").value == "first"


def test_start_end_position_untokenized() -> None:
    sentence: Sentence = Sentence("This is a sentence.", start_position=10)
    assert sentence.start_position == 10
    assert sentence.end_position == 29
    assert [(token.start_position, token.end_position) for token in sentence] == [
        (0, 4),
        (5, 7),
        (8, 9),
        (10, 18),
        (18, 19),
    ]


def test_start_end_position_pretokenized() -> None:
    # Initializing a Sentence this way assumes that there is a space after each token
    sentence: Sentence = Sentence(["This", "is", "a", "sentence", "."], start_position=10)
    assert sentence.start_position == 10
    assert sentence.end_position == 30
    assert [(token.start_position, token.end_position) for token in sentence] == [
        (0, 4),
        (5, 7),
        (8, 9),
        (10, 18),
        (19, 20),
    ]
