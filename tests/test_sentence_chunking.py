from typing import Dict, List

import pytest

from flair.data import Sentence
from flair.training_utils import CharEntity, create_flair_sentence


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
def small_token_limit_resume() -> Dict:
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
def small_token_limit_response() -> List[Sentence]:
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
        sentences = create_flair_sentence("", [])
        assert len(sentences) == 0

    def check_split_entities(self, entity_labels, chunks, max_token_limit):
        """Ensure that no entities are split over chunks (except entities longer than the token limit)."""
        chunk_intervals = []
        start_index = 0
        for chunk in chunks:
            end_index = start_index + len(chunk.text)
            chunk_intervals.append((start_index, end_index))
            start_index = end_index

        for entity in entity_labels:
            entity_start, entity_end = entity.start_char_idx, entity.end_char_idx
            entity_length = entity_end - entity_start

            # Skip the check if the entity itself is longer than the maximum token limit
            if entity_length > max_token_limit:
                continue

            assert any(
                start <= entity_start and entity_end <= end for start, end in chunk_intervals
            ), f"Entity {entity} is not within a single chunk interval"

    @pytest.mark.parametrize(
        "test_text, expected_text",
        [
            ("test text", "test text"),
            ("a", "a"),
            ("this ", "this"),
        ],
    )
    def test_short_text(self, test_text, expected_text):
        """Short texts that should fit nicely into a single chunk."""
        chunks = create_flair_sentence(test_text, [])
        assert chunks[0].text == expected_text

    def test_create_flair_sentence(self, parsed_resume_dict):
        chunks = create_flair_sentence(parsed_resume_dict["raw_text"], parsed_resume_dict["entities"])
        assert len(chunks) == 2

        max_token_limit = 512  # default
        assert all(len(c) <= max_token_limit for c in chunks)

        self.check_split_entities(parsed_resume_dict["entities"], chunks, max_token_limit)

    def test_small_token_limit(self, small_token_limit_resume, small_token_limit_response):
        max_token_limit = 10  # test a small max token limit
        chunks = create_flair_sentence(
            small_token_limit_resume["raw_text"], small_token_limit_resume["entities"], token_limit=max_token_limit
        )

        for response, expected in zip(chunks, small_token_limit_response):
            assert response.to_tagged_string() == expected.to_tagged_string()

        assert all(len(c) <= max_token_limit for c in chunks)

        self.check_split_entities(small_token_limit_resume["entities"], chunks, max_token_limit)

    @pytest.mark.parametrize(
        "test_text, entities, expected_chunks",
        [
            (
                "Led a team of five engineers. It's important to note the project's success. We've implemented state-of-the-art technologies. Co-ordinated efforts with cross-functional teams.",
                [
                    CharEntity(0, 25, "RESPONSIBILITY", "Led a team of five engineers"),
                    CharEntity(27, 72, "ACHIEVEMENT", "It's important to note the project's success"),
                    CharEntity(74, 117, "ACHIEVEMENT", "We've implemented state-of-the-art technologies"),
                    CharEntity(119, 168, "RESPONSIBILITY", "Co-ordinated efforts with cross-functional teams"),
                ],
                [
                    "Led a team of five engine er s. It 's important to note the project 's succe ss",
                    ". We 've implemented state-of-the-art techno lo gies . Co-ordinated efforts with cross-functional teams .",
                ],
            ),
        ],
    )
    def test_contractions_and_hyphens(self, test_text, entities, expected_chunks):
        max_token_limit = 20
        chunks = create_flair_sentence(test_text, entities, max_token_limit)
        for i, chunk in enumerate(expected_chunks):
            assert chunks[i].text == chunk
        self.check_split_entities(entities, chunks, max_token_limit)

    @pytest.mark.parametrize(
        "test_text, entities",
        [
            (
                "This is a long text. " * 100,
                [CharEntity(0, 1000, "dummy_label1", "Dummy Text 1")],
            )
        ],
    )
    def test_long_text(self, test_text, entities):
        """Test for handling long texts that should be split into multiple chunks."""
        max_token_limit = 512
        chunks = create_flair_sentence(test_text, entities, max_token_limit)
        assert len(chunks) > 1
        assert all(len(c) <= max_token_limit for c in chunks)
        self.check_split_entities(entities, chunks, max_token_limit)

    @pytest.mark.parametrize(
        "test_text, entities, expected_chunks",
        [
            (
                "Hello! Is your company hiring? I am available for employment. Contact me at 5:00 p.m.",
                [
                    CharEntity(0, 6, "LABEL", "Hello!"),
                    CharEntity(7, 31, "LABEL", "Is your company hiring?"),
                    CharEntity(32, 65, "LABEL", "I am available for employment."),
                    CharEntity(66, 86, "LABEL", "Contact me at 5:00 p.m."),
                ],
                [
                    "Hello ! Is your company hiring ? I",
                    "am available for employment . Con t",
                    "act me at 5:00 p.m .",
                ],
            )
        ],
    )
    def test_text_with_punctuation(self, test_text, entities, expected_chunks):
        max_token_limit = 10
        chunks = create_flair_sentence(test_text, entities, max_token_limit)
        for i, chunk in enumerate(expected_chunks):
            assert chunks[i].text == chunk
        self.check_split_entities(entities, chunks, max_token_limit)
