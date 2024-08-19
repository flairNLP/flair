import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from flair.datasets.biomedical import (
    CoNLLWriter,
    Entity,
    InternalBioNerDataset,
    filter_nested_entities,
)
from flair.splitter import NoSentenceSplitter, SentenceSplitter
from flair.tokenization import SpaceTokenizer

logger = logging.getLogger("flair")
logger.propagate = True


def test_write_to_conll():
    text = "This is entity1 entity2 and a long entity3"
    dataset = InternalBioNerDataset(
        documents={"1": text},
        entities_per_document={
            "1": [
                Entity((text.find("entity1"), text.find("entity1") + len("entity1")), "E"),
                Entity((text.find("entity2"), text.find("entity2") + len("entity2")), "E"),
                Entity(
                    (
                        text.find("a long entity3"),
                        text.find("a long entity3") + len("a long entity3"),
                    ),
                    "E",
                ),
            ]
        },
        entity_types=["E"],
    )
    expected_labeling = [
        "This O +",
        "is O +",
        "entity1 B-E +",
        "entity2 B-E +",
        "and O +",
        "a B-E +",
        "long I-E +",
        "entity3 I-E -",
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
                Entity((text.find("entity2"), text.find("entity2") + len("entity2")), "E"),
            ]
        },
        entity_types=["E"],
    )

    assert_conll_writer_output(dataset, ["This O +", "is O +", "entity1 B-E +", "entity2 B-E -"])


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
        entity_types=["E"],
    )

    assert_conll_writer_output(dataset, ["This O +", "is O +", "entity1 B-E +", "entity2 O -"])


def assert_conll_writer_output(
    dataset: InternalBioNerDataset,
    expected_output: List[str],
    sentence_splitter: Optional[SentenceSplitter] = None,
):
    fd, outfile_path = tempfile.mkstemp()
    try:
        sentence_splitter = sentence_splitter if sentence_splitter else NoSentenceSplitter(tokenizer=SpaceTokenizer())

        writer = CoNLLWriter(sentence_splitter=sentence_splitter)
        writer.write_to_conll(dataset, Path(outfile_path))
        with open(outfile_path) as f:
            contents = [line.strip() for line in f.readlines() if line.strip()]
    finally:
        os.close(fd)
        os.remove(outfile_path)

    assert contents == expected_output


def test_filter_nested_entities(caplog):
    entities_per_document = {
        "d0": [Entity((0, 1), "t0"), Entity((2, 3), "t1")],
        "d1": [Entity((0, 6), "t0"), Entity((2, 3), "t1"), Entity((4, 5), "t2")],
        "d2": [Entity((0, 3), "t0"), Entity((3, 5), "t1")],
        "d3": [Entity((0, 3), "t0"), Entity((2, 5), "t1"), Entity((4, 7), "t2")],
        "d4": [Entity((0, 4), "t0"), Entity((3, 5), "t1")],
        "d5": [Entity((0, 4), "t0"), Entity((3, 9), "t1")],
        "d6": [Entity((0, 4), "t0"), Entity((2, 6), "t1")],
    }

    target = {
        "d0": [Entity((0, 1), "t0"), Entity((2, 3), "t1")],
        "d1": [Entity((2, 3), "t1"), Entity((4, 5), "t2")],
        "d2": [Entity((0, 3), "t0"), Entity((3, 5), "t1")],
        "d3": [Entity((0, 3), "t0"), Entity((4, 7), "t2")],
        "d4": [Entity((0, 4), "t0")],
        "d5": [Entity((3, 9), "t1")],
        "d6": [Entity((0, 4), "t0")],
    }

    dataset = InternalBioNerDataset(documents={}, entities_per_document=entities_per_document)
    caplog.set_level(logging.WARNING)
    filter_nested_entities(dataset)

    assert "WARNING: Corpus modified by filtering nested entities." in caplog.text

    for key, entities in dataset.entities_per_document.items():
        assert key in target
        assert len(target[key]) == len(entities)
        for e1, e2 in zip(
            sorted(target[key], key=lambda x: str(x)),
            sorted(entities, key=lambda x: str(x)),
        ):
            assert str(e1) == str(e2)
