from typing import cast

import pytest

from flair.data import Sentence
from flair.training_utils import CharEntity, TokenEntity, create_labeled_sentence_from_entity_offsets


@pytest.fixture(params=["resume1.txt"])
def resume(request, resources_path) -> str:
    filepath = resources_path / "text_sequences" / request.param
    with open(filepath, encoding="utf8") as file:
        text_content = file.read()
    return text_content


@pytest.fixture
def parsed_resume_dict(resume) -> dict:
    return {
        "raw_text": resume,
        "entities": [
            CharEntity(20, 40, "dummy_label1", "Dummy Text 1"),
            CharEntity(250, 300, "dummy_label2", "Dummy Text 2"),
            CharEntity(700, 810, "dummy_label3", "Dummy Text 3"),
            CharEntity(3900, 4000, "dummy_label4", "Dummy Text 4"),
        ],
    }


@pytest.fixture
def small_token_limit_resume() -> dict:
    return {
        "raw_text": "Professional Clown June 2020 - August 2021 Entertaining students of all ages. Blah Blah Blah "
        "Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah "
        "Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah "
        "Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah "
        "Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Blah Gained "
        "proficiency in juggling and scaring children.",
        "entities": [
            CharEntity(0, 18, "EXPERIENCE.TITLE", ""),
            CharEntity(19, 29, "DATE.START_DATE", ""),
            CharEntity(31, 42, "DATE.END_DATE", ""),
            CharEntity(450, 510, "EXPERIENCE.DESCRIPTION", ""),
        ],
    }


@pytest.fixture
def small_token_limit_response() -> list[Sentence]:
    """Recreates expected response Sentences."""
    chunk0 = Sentence("Professional Clown June 2020 - August 2021 Entertaining students of")
    chunk0[0:2].add_label("Professional Clown", "EXPERIENCE.TITLE")
    chunk0[2:4].add_label("June 2020", "DATE.START_DATE")
    chunk0[5:7].add_label("August 2021", "DATE.END_DATE")

    chunk1 = Sentence("Blah Blah Blah Blah Blah Blah Blah Bl")

    chunk2 = Sentence("ah Blah Gained proficiency in juggling and scaring children .")
    chunk2[0:10].add_label("ah Blah Gained proficiency in juggling and scaring children .", "EXPERIENCE.DESCRIPTION")

    return [chunk0, chunk1, chunk2]


class TestChunking:
    def test_empty_string(self):
        sentences = create_labeled_sentence_from_entity_offsets("", [])
        assert len(sentences) == 0

    def check_tokens(self, sentence: Sentence, expected_tokens: list[str]):
        assert len(sentence.tokens) == len(expected_tokens)
        assert [token.text for token in sentence.tokens] == expected_tokens
        for token, expected_token in zip(sentence.tokens, expected_tokens):
            assert token.text == expected_token

    def check_token_entities(self, sentence: Sentence, expected_labels: list[TokenEntity]):
        assert len(sentence.labels) == len(expected_labels)
        for label, expected_label in zip(sentence.labels, expected_labels):

            assert label.value == expected_label.label
            span = cast(Sentence, label.data_point)
            assert span.tokens[0]._internal_index is not None
            assert span.tokens[0]._internal_index - 1 == expected_label.start_token_idx
            assert span.tokens[-1]._internal_index is not None
            assert span.tokens[-1]._internal_index - 1 == expected_label.end_token_idx

    def check_split_entities(self, entity_labels, sentence: Sentence):
        """Ensure that no entities are split over chunks (except entities longer than the token limit)."""
        for entity in entity_labels:
            entity_start, entity_end = entity.start_char_idx, entity.end_char_idx
            assert entity_start >= 0 and entity_end <= len(
                sentence
            ), f"Entity {entity} is not within a single chunk interval"

    @pytest.mark.parametrize(
        "test_text, expected_text",
        [
            ("test text", "test text"),
            ("a", "a"),
            ("this ", "this"),
        ],
    )
    def test_short_text(self, test_text: str, expected_text: str):
        """Short texts that should fit nicely into a single chunk."""
        chunks = create_labeled_sentence_from_entity_offsets(test_text, [])
        assert chunks.text == expected_text

    def test_create_labeled_sentence(self, parsed_resume_dict: dict):
        create_labeled_sentence_from_entity_offsets(parsed_resume_dict["raw_text"], parsed_resume_dict["entities"])

    @pytest.mark.parametrize(
        "test_text, entities, expected_tokens, expected_labels",
        [
            (
                "Led a team of five engineers. It's important to note the project's success. We've implemented state-of-the-art technologies. Co-ordinated efforts with cross-functional teams.",
                [
                    CharEntity(0, 28, "RESPONSIBILITY", "Led a team of five engineers"),
                    CharEntity(30, 74, "ACHIEVEMENT", "It's important to note the project's success"),
                    CharEntity(76, 123, "ACHIEVEMENT", "We've implemented state-of-the-art technologies"),
                    CharEntity(125, 173, "RESPONSIBILITY", "Co-ordinated efforts with cross-functional teams"),
                ],
                [
                    "Led",
                    "a",
                    "team",
                    "of",
                    "five",
                    "engineers",
                    ".",
                    "It",
                    "'s",
                    "important",
                    "to",
                    "note",
                    "the",
                    "project",
                    "'s",
                    "success",
                    ".",
                    "We",
                    "'ve",
                    "implemented",
                    "state-of-the-art",
                    "technologies",
                    ".",
                    "Co-ordinated",
                    "efforts",
                    "with",
                    "cross-functional",
                    "teams",
                    ".",
                ],
                [
                    TokenEntity(0, 5, "RESPONSIBILITY"),
                    TokenEntity(7, 15, "ACHIEVEMENT"),
                    TokenEntity(17, 21, "ACHIEVEMENT"),
                    TokenEntity(23, 27, "RESPONSIBILITY"),
                ],
            ),
        ],
    )
    def test_contractions_and_hyphens(
        self, test_text: str, entities: list[CharEntity], expected_tokens: list[str], expected_labels: list[TokenEntity]
    ):
        sentence = create_labeled_sentence_from_entity_offsets(test_text, entities)
        self.check_tokens(sentence, expected_tokens)
        self.check_token_entities(sentence, expected_labels)

    @pytest.mark.parametrize(
        "test_text, entities",
        [
            (
                "This is a long text. " * 100,
                [CharEntity(0, 1000, "dummy_label1", "Dummy Text 1")],
            )
        ],
    )
    def test_long_text(self, test_text: str, entities: list[CharEntity]):
        """Test for handling long texts that should be split into multiple chunks."""
        create_labeled_sentence_from_entity_offsets(test_text, entities)

    @pytest.mark.parametrize(
        "test_text, entities, expected_labels",
        [
            (
                "Hello! Is your company hiring? I am available for employment. Contact me at 5:00 p.m.",
                [
                    CharEntity(0, 6, "LABEL", "Hello!"),
                    CharEntity(7, 30, "LABEL", "Is your company hiring?"),
                    CharEntity(31, 61, "LABEL", "I am available for employment."),
                    CharEntity(62, 85, "LABEL", "Contact me at 5:00 p.m."),
                ],
                [
                    TokenEntity(0, 1, "LABEL"),
                    TokenEntity(2, 6, "LABEL"),
                    TokenEntity(7, 12, "LABEL"),
                    TokenEntity(13, 18, "LABEL"),
                ],
            )
        ],
    )
    def test_text_with_punctuation(
        self, test_text: str, entities: list[CharEntity], expected_labels: list[TokenEntity]
    ):
        sentence = create_labeled_sentence_from_entity_offsets(test_text, entities)
        self.check_token_entities(sentence, expected_labels)
