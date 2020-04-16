import os
import tempfile

from pathlib import Path
from typing import List

from flair.datasets.biomedical import (
    Entity,
    InternalBioNerDataset,
    whitespace_tokenize,
    CoNLLWriter,
    find_overlapping_entities,
    find_nested_entities,
    normalize_entity_spans,
)


def test_find_overlapping_entities():
    not_overlapping_entities = [Entity((0, 3), ""), Entity((5, 10), "")]
    result = find_overlapping_entities(not_overlapping_entities)
    assert len(result) == 0

    not_overlapping_entities = [Entity((0, 3), ""), Entity((3, 6), "")]
    result = find_overlapping_entities(not_overlapping_entities)
    assert len(result) == 0

    entity1 = Entity((0, 6), "")
    entity2 = Entity((3, 7), "")
    entity3 = Entity((10, 12), "")
    simple_overlapping_entities = [entity1, entity2, entity3]
    result = find_overlapping_entities(simple_overlapping_entities)
    assert len(result) == 1
    assert result[0] == (entity1, entity2)

    entity1 = Entity((0, 6), "")
    entity2 = Entity((3, 6), "")
    entity3 = Entity((4, 8), "")
    entity4 = Entity((7, 10), "")
    entity5 = Entity((12, 13), "")
    simple_overlapping_entities = [entity1, entity2, entity3, entity4, entity5]
    result = find_overlapping_entities(simple_overlapping_entities)
    assert len(result) == 4
    assert result[0] == (entity1, entity2)
    assert result[1] == (entity1, entity3)
    assert result[2] == (entity2, entity3)
    assert result[3] == (entity3, entity4)


def test_find_nested_entities():
    entity1 = Entity((0, 12), "a")
    entity2 = Entity((1, 4), "a")
    entity3 = Entity((5, 6), "a")

    result = find_nested_entities([entity3, entity1, entity2])
    assert len(result) == 1
    assert result[0].char_span == range(0, 12)
    assert result[0].type == "a"
    assert len(result[0].nested_entities) == 2
    assert entity2 in result[0].nested_entities
    assert entity3 in result[0].nested_entities


def test_normalize_entity_spans():
    entity1 = Entity((0, 12), "")
    entity2 = Entity((1, 4), "")
    entity3 = Entity((5, 6), "")

    result = normalize_entity_spans([entity2, entity1, entity3])
    assert len(result) == 2
    assert result[0] == entity2
    assert result[1] == entity3

    entity1 = Entity((0, 20), "")
    entity2 = Entity((0, 10), "")
    entity3 = Entity((10, 20), "")
    entity4 = Entity((2, 8), "")
    entity5 = Entity((12, 18), "")

    result = normalize_entity_spans([entity2, entity5, entity1, entity3, entity4])
    assert len(result) == 2
    assert result[0] == entity2
    assert result[1] == entity3

    entity1 = Entity((0, 12), "")
    entity2 = Entity((1, 4), "")
    entity3 = Entity((5, 6), "")
    entity4 = Entity((8, 16), "")

    result = normalize_entity_spans([entity2, entity1, entity4, entity3])
    assert len(result) == 3
    assert result[0] == entity2
    assert result[1] == entity3
    assert result[2].char_span == range(12, 16)


def test_write_to_conll():
    text = "This is entity1 entity2 and a long entity3"
    dataset = InternalBioNerDataset(
        documents={"1": text},
        entities_per_document={
            "1": [
                Entity(
                    (text.find("entity1"), text.find("entity1") + len("entity1")), "E"
                ),
                Entity(
                    (text.find("entity2"), text.find("entity2") + len("entity2")), "E"
                ),
                Entity(
                    (
                        text.find("a long entity3"),
                        text.find("a long entity3") + len("a long entity3"),
                    ),
                    "E",
                ),
            ]
        },
    )
    expected_labeling = [
        "This O",
        "is O",
        "entity1 B-E",
        "entity2 B-E",
        "and O",
        "a B-E",
        "long I-E",
        "entity3 I-E",
    ]
    assert_conll_writer_output(dataset, expected_labeling)


def test_conll_writer_one_token_multiple_entities1():
    text = "This is entity1 entity2"
    dataset = InternalBioNerDataset(
        documents={"1": text},
        entities_per_document={
            "1": [
                Entity((text.find("entity1"), text.find("entity1") + 2), "E"),
                Entity((text.find("tity1"), text.find("tity1") + 5), "E"),
                Entity(
                    (text.find("entity2"), text.find("entity2") + len("entity2")), "E"
                ),
            ]
        },
    )

    assert_conll_writer_output(
        dataset, ["This O", "is O", "entity1 B-E", "entity2 B-E"]
    )


def test_conll_writer_one_token_multiple_entities2():
    text = "This is entity1 entity2"
    dataset = InternalBioNerDataset(
        documents={"1": text},
        entities_per_document={
            "1": [
                Entity((text.find("entity1"), text.find("entity1") + 2), "E"),
                Entity((text.find("tity1"), text.find("tity1") + 5), "E"),
            ]
        },
    )

    assert_conll_writer_output(dataset, ["This O", "is O", "entity1 B-E", "entity2 O"])


def assert_conll_writer_output(
    dataset: InternalBioNerDataset, expected_output: List[str]
):
    outfile_path = tempfile.mkstemp()[1]
    try:
        writer = CoNLLWriter(
            tokenizer=whitespace_tokenize, sentence_splitter=lambda x: ([x], [0])
        )
        writer.write_to_conll(dataset, Path(outfile_path))
        contents = [l.strip() for l in open(outfile_path).readlines() if l.strip()]
    finally:
        os.remove(outfile_path)

    assert contents == expected_output


def test_whitespace_tokenizer():
    tokens, offsets = whitespace_tokenize("Abc def .")
    assert tokens == ["Abc", "def", "."]
    assert offsets == [0, 4, 8]

    tokens, offsets = whitespace_tokenize("Abc Abc .")
    assert tokens == ["Abc", "Abc", "."]
    assert offsets == [0, 4, 8]
