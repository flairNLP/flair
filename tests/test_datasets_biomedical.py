import inspect
import os
import tempfile
from operator import itemgetter
from pathlib import Path
from typing import Callable, List, Type

import pytest
from tqdm import tqdm  # type: ignore

import flair
import flair.datasets.biomedical as biomedical
from flair.data import Sentence, Token, _iter_dataset
from flair.datasets import ColumnCorpus
from flair.datasets.biomedical import (
    CoNLLWriter,
    Entity,
    HunerDataset,
    InternalBioNerDataset,
    filter_nested_entities,
)
from flair.tokenization import NoSentenceSplitter, SentenceSplitter, SpaceTokenizer


def has_balanced_parantheses(text: str) -> bool:
    stack = []
    opening = ["(", "[", "{"]
    closing = [")", "]", "}"]
    for c in text:
        if c in opening:
            stack.append(c)
        elif c in closing:
            if not stack:
                return False
            last_paren = stack.pop()
            if opening.index(last_paren) != closing.index(c):
                return False

    return len(stack) == 0


def gene_predicate(member):
    return "HUNER_GENE_" in str(member) and inspect.isclass(member)


def chemical_predicate(member):
    return "HUNER_CHEMICAL_" in str(member) and inspect.isclass(member)


def disease_predicate(member):
    return "HUNER_DISEASE_" in str(member) and inspect.isclass(member)


def species_predicate(member):
    return "HUNER_SPECIES_" in str(member) and inspect.isclass(member)


def cellline_predicate(member):
    return "HUNER_CELL_LINE_" in str(member) and inspect.isclass(member)


CELLLINE_DATASETS = [
    i[1] for i in sorted(inspect.getmembers(biomedical, predicate=cellline_predicate), key=itemgetter(0))
]
CHEMICAL_DATASETS = [
    i[1] for i in sorted(inspect.getmembers(biomedical, predicate=chemical_predicate), key=itemgetter(0))
]
DISEASE_DATASETS = [
    i[1] for i in sorted(inspect.getmembers(biomedical, predicate=disease_predicate), key=itemgetter(0))
]
GENE_DATASETS = [i[1] for i in sorted(inspect.getmembers(biomedical, predicate=gene_predicate), key=itemgetter(0))]
SPECIES_DATASETS = [
    i[1] for i in sorted(inspect.getmembers(biomedical, predicate=species_predicate), key=itemgetter(0))
]
ALL_DATASETS = CELLLINE_DATASETS + CHEMICAL_DATASETS + DISEASE_DATASETS + GENE_DATASETS + SPECIES_DATASETS


def simple_tokenizer(text: str) -> List[str]:
    tokens: List[str] = []
    word = ""
    index = -1
    for index, char in enumerate(text):
        if char == " " or char == "-":
            if len(word) > 0:
                tokens.append(word)

            word = ""
        else:
            word += char

    # increment for last token in sentence if not followed by whitespace
    index += 1
    if len(word) > 0:
        tokens.append(word)

    return tokens


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
    )

    assert_conll_writer_output(dataset, ["This O +", "is O +", "entity1 B-E +", "entity2 O -"])


def assert_conll_writer_output(
    dataset: InternalBioNerDataset,
    expected_output: List[str],
    sentence_splitter: SentenceSplitter = None,
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


def test_filter_nested_entities():
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
    with pytest.warns(UserWarning):
        filter_nested_entities(dataset)

    for key, entities in dataset.entities_per_document.items():
        assert key in target
        assert len(target[key]) == len(entities)
        for e1, e2 in zip(
            sorted(target[key], key=lambda x: str(x)),
            sorted(entities, key=lambda x: str(x)),
        ):
            assert str(e1) == str(e2)


def sanity_check_all_corpora(check: Callable[[ColumnCorpus], None]):
    for _, CorpusType in tqdm(ALL_DATASETS):
        corpus = CorpusType()
        check(corpus)


@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
def test_sanity_not_starting_with_minus(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()  # type: ignore
    entities_starting_with_minus = []
    for sentence in _iter_dataset(corpus.get_all_sentences()):
        entities = sentence.get_spans("ner")
        for entity in entities:
            if str(entity.tokens[0].text).startswith("-"):
                entities_starting_with_minus.append(" ".join([t.text for t in entity.tokens]))

    assert len(entities_starting_with_minus) == 0, "|".join(entities_starting_with_minus)


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_sanity_no_repeating_Bs(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()  # type: ignore
    longest_repeat_tokens: List[Token] = []
    repeat_tokens: List[Token] = []
    for sentence in _iter_dataset(corpus.get_all_sentences()):
        for token in sentence.tokens:
            if token.get_labels()[0].value.startswith("B") or token.get_labels()[0].value.startswith("S"):
                repeat_tokens.append(token)
            else:
                if len(repeat_tokens) > len(longest_repeat_tokens):
                    longest_repeat_tokens = repeat_tokens
                repeat_tokens = []

    assert len(longest_repeat_tokens) < 4


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_sanity_no_long_entities(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()  # type: ignore
    longest_entity: List[str] = []
    for sentence in _iter_dataset(corpus.get_all_sentences()):
        entities = sentence.get_spans("ner")
        for entity in entities:
            if len(entity.tokens) > len(longest_entity):
                longest_entity = [t.text for t in entity.tokens]

    assert len(longest_entity) < 10, " ".join(longest_entity)


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_sanity_no_unmatched_parentheses(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()  # type: ignore
    unbalanced_entities = []
    for sentence in _iter_dataset(corpus.get_all_sentences()):
        entities = sentence.get_spans("ner")
        for entity in entities:
            entity_text = "".join(t.text for t in entity.tokens)
            if not has_balanced_parantheses(entity_text):
                unbalanced_entities.append(entity_text)

    assert unbalanced_entities == []


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_sanity_not_too_many_entities(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()  # type: ignore
    n_entities_per_sentence = []
    for sentence in _iter_dataset(corpus.get_all_sentences()):
        entities = sentence.get_spans("ner")
        n_entities_per_sentence.append(len(entities))
    avg_entities_per_sentence = sum(n_entities_per_sentence) / len(n_entities_per_sentence)

    assert avg_entities_per_sentence <= 5


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_sanity_no_misaligned_entities(CorpusType: Type[HunerDataset]):
    dataset_name = CorpusType.__class__.__name__.lower()
    base_path = flair.cache_root / "datasets"
    data_folder = base_path / dataset_name

    corpus = CorpusType()
    internal = corpus.to_internal(data_folder)
    for doc_id, doc_text in internal.documents.items():
        misaligned_starts = []
        misaligned_ends: List[int] = []

        entities = internal.entities_per_document[doc_id]
        entity_starts = [i.char_span.start for i in entities]
        entity_ends = [i.char_span.stop for i in entities]

        for start in entity_starts:
            if start not in entity_starts:
                misaligned_starts.append(start)

        for end in entity_ends:
            if end not in entity_ends:
                misaligned_starts.append(end)

        assert len(misaligned_starts) <= len(entities) // 10
        assert len(misaligned_ends) <= len(entities) // 10


@pytest.mark.skip(reason="We skip this test because it's only relevant for development purposes")
def test_scispacy_tokenization():
    from flair.tokenization import SciSpacyTokenizer

    spacy_tokenizer = SciSpacyTokenizer()

    sentence = Sentence("HBeAg(+) patients", use_tokenizer=spacy_tokenizer)
    assert len(sentence) == 5
    assert sentence[0].text == "HBeAg"
    assert sentence[0].start_position == 0
    assert sentence[1].text == "("
    assert sentence[1].start_position == 5
    assert sentence[2].text == "+"
    assert sentence[2].start_position == 6
    assert sentence[3].text == ")"
    assert sentence[3].start_position == 7
    assert sentence[4].text == "patients"
    assert sentence[4].start_position == 9

    sentence = Sentence("HBeAg(+)/HBsAg(+)", use_tokenizer=spacy_tokenizer)
    assert len(sentence) == 9

    assert sentence[0].text == "HBeAg"
    assert sentence[0].start_position == 0
    assert sentence[1].text == "("
    assert sentence[1].start_position == 5
    assert sentence[2].text == "+"
    assert sentence[2].start_position == 6
    assert sentence[3].text == ")"
    assert sentence[3].start_position == 7
    assert sentence[4].text == "/"
    assert sentence[4].start_position == 8
    assert sentence[5].text == "HBsAg"
    assert sentence[5].start_position == 9
    assert sentence[6].text == "("
    assert sentence[6].start_position == 14
    assert sentence[7].text == "+"
    assert sentence[7].start_position == 15
    assert sentence[8].text == ")"
    assert sentence[8].start_position == 16

    sentence = Sentence("doxorubicin (DOX)-induced", use_tokenizer=spacy_tokenizer)

    assert len(sentence) == 5
    assert sentence[0].text == "doxorubicin"
    assert sentence[1].text == "("
    assert sentence[2].text == "DOX"
    assert sentence[3].text == ")"
    assert sentence[4].text == "-induced"
