import copy
from flair.data import Sentence, Relation
from flair.tokenization import SegtokTokenizer, SpaceTokenizer


def _create_annotated_sentence() -> Sentence:
    """
    Creates a complexly annotated sentence for testing.
    "George Washington went to Washington."
    - Sentence-level label
    - "George Washington" is a PERSON
    - "Washington" is a LOCATION
    - Relation: (George Washington)-[went_to]->(Washington)
    """
    sentence = Sentence(
        "George Washington went to Washington.",
        use_tokenizer=SegtokTokenizer(),
    )

    # Add sentence-level label
    sentence.add_label("category", "HISTORICAL_EVENT")

    # Add span labels
    sentence[0:2].add_label("ner", "PERSON")  # "George Washington"
    sentence[4:5].add_label("ner", "LOCATION")  # "Washington"

    # Add relation label
    person_span = sentence[0:2]
    location_span = sentence[4:5]

    relation = Relation(person_span, location_span)
    relation.add_label("relation", "WENT_TO")

    return sentence


def _assert_sentences_equal(s1: Sentence, s2: Sentence):
    """Helper method to assert equality between two sentences."""
    # 1. Check basic properties
    assert s1.to_original_text() == s2.to_original_text()
    assert s1.tokenizer.name == s2.tokenizer.name
    assert len(s1) == len(s2)

    # 2. Check tokens
    for t1, t2 in zip(s1.tokens, s2.tokens):
        assert t1.text == t2.text
        assert t1.start_position == t2.start_position

    # 3. Check sentence-level labels
    s1_sent_labels = {(label.value, label.score) for label in s1.get_labels("category")}
    s2_sent_labels = {(label.value, label.score) for label in s2.get_labels("category")}
    assert s1_sent_labels == s2_sent_labels

    # 4. Check spans
    s1_spans = {(span.text, span.tag) for span in s1.get_spans("ner")}
    s2_spans = {(span.text, span.tag) for span in s2.get_spans("ner")}
    assert s1_spans == s2_spans
    assert len(s1.get_spans("ner")) == len(s2.get_spans("ner"))

    # 5. Check relations
    s1_relations = {label.value for label in s1.get_labels("relation")}
    s2_relations = {label.value for label in s2.get_labels("relation")}
    assert s1_relations == s2_relations


def test_deepcopy_preserves_annotations():
    """Tests that copy.deepcopy() creates a true, independent copy."""
    original_sentence = _create_annotated_sentence()

    # Perform deepcopy
    copied_sentence = copy.deepcopy(original_sentence)

    # Assert they are different objects in memory
    assert id(original_sentence) != id(copied_sentence)
    assert id(original_sentence.tokens[0]) != id(copied_sentence.tokens[0])

    # Assert content is identical
    _assert_sentences_equal(original_sentence, copied_sentence)


def test_json_serialization_preserves_annotations():
    """Tests the to_dict() -> from_dict() cycle."""
    original_sentence = _create_annotated_sentence()

    # Serialize and deserialize
    sentence_dict = original_sentence.to_dict()
    recreated_sentence = Sentence.from_dict(sentence_dict)

    # Assert they are different objects in memory
    assert id(original_sentence) != id(recreated_sentence)

    # Assert content is identical
    _assert_sentences_equal(
        original_sentence,
        recreated_sentence,
    )


def test_serialization_preserves_tokenizer():
    """Tests that a non-default tokenizer is preserved."""
    # Use a non-default tokenizer
    sentence = Sentence("A simple test.", use_tokenizer=SpaceTokenizer())
    sentence.add_label("topic", "testing")

    # Serialize and deserialize
    recreated_sentence = Sentence.from_dict(sentence.to_dict())

    # Check that the tokenizer is of the correct type
    assert isinstance(recreated_sentence.tokenizer, SpaceTokenizer)
    assert sentence.tokenizer.name == recreated_sentence.tokenizer.name
    assert len(sentence.tokens) == len(recreated_sentence.tokens)
