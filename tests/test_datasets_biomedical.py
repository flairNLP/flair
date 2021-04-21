import inspect
import flair
import os
import tempfile
import pytest

import flair.datasets.biomedical as biomedical

from operator import itemgetter
from pathlib import Path
from typing import List, Callable, Type
from tqdm import tqdm

from flair.tokenization import (
    TokenizerWrapper,
    SpaceTokenizer,
    TagSentenceSplitter,
    SentenceSplitter,
    NoSentenceSplitter
)

from flair.data import Token
from flair.datasets import ColumnCorpus
from flair.datasets.biomedical import (
    Entity,
    InternalBioNerDataset,
    CoNLLWriter,
    filter_nested_entities,
    SENTENCE_TAG,
    HunerDataset,
)


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
    i[1]
    for i in sorted(
        inspect.getmembers(biomedical, predicate=cellline_predicate), key=itemgetter(0)
    )
]
CHEMICAL_DATASETS = [
    i[1]
    for i in sorted(
        inspect.getmembers(biomedical, predicate=chemical_predicate), key=itemgetter(0)
    )
]
DISEASE_DATASETS = [
    i[1]
    for i in sorted(
        inspect.getmembers(biomedical, predicate=disease_predicate), key=itemgetter(0)
    )
]
GENE_DATASETS = [
    i[1]
    for i in sorted(
        inspect.getmembers(biomedical, predicate=gene_predicate), key=itemgetter(0)
    )
]
SPECIES_DATASETS = [
    i[1]
    for i in sorted(
        inspect.getmembers(biomedical, predicate=species_predicate), key=itemgetter(0)
    )
]
ALL_DATASETS = (
    CELLLINE_DATASETS
    + CHEMICAL_DATASETS
    + DISEASE_DATASETS
    + GENE_DATASETS
    + SPECIES_DATASETS
)


def simple_tokenizer(text: str) -> List[Token]:
    tokens: List[Token] = []
    word = ""
    index = -1
    for index, char in enumerate(text):
        if char == " " or char == "-":
            if len(word) > 0:
                start_position = index - len(word)
                tokens.append(
                    Token(
                        text=word, start_position=start_position, whitespace_after=(char == " ")
                    )
                )

            word = ""
        else:
            word += char

    # increment for last token in sentence if not followed by whitespace
    index += 1
    if len(word) > 0:
        start_position = index - len(word)
        tokens.append(
            Token(text=word, start_position=start_position, whitespace_after=False)
        )

    return tokens


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
                Entity(
                    (text.find("entity2"), text.find("entity2") + len("entity2")), "E"
                ),
            ]
        },
    )

    assert_conll_writer_output(
        dataset, ["This O +", "is O +", "entity1 B-E +", "entity2 B-E -"]
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

    assert_conll_writer_output(
        dataset, ["This O +", "is O +", "entity1 B-E +", "entity2 O -"]
    )


def test_conll_writer_whitespace_after():
    text = f"A sentence with cardio-dependent. {SENTENCE_TAG}Clark et al. reported that"
    dataset = InternalBioNerDataset(
        documents={"1": text}, entities_per_document={"1": []},
    )

    assert_conll_writer_output(
        dataset,
        [
            "A O +",
            "sentence O +",
            "with O +",
            "cardio O -",
            "dependent. O +",
            "Clark O +",
            "et O +",
            "al. O +",
            "reported O +",
            "that O -",
        ],
        TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=TokenizerWrapper(simple_tokenizer))
    )


def assert_conll_writer_output(
    dataset: InternalBioNerDataset,
    expected_output: List[str],
    sentence_splitter: SentenceSplitter = None,
):
    outfile_path = tempfile.mkstemp()[1]
    try:
        sentence_splitter = (
            sentence_splitter if sentence_splitter else NoSentenceSplitter(tokenizer=SpaceTokenizer())
        )

        writer = CoNLLWriter(sentence_splitter=sentence_splitter)
        writer.write_to_conll(dataset, Path(outfile_path))
        contents = [l.strip() for l in open(outfile_path).readlines() if l.strip()]

    finally:
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

    dataset = InternalBioNerDataset(
        documents={}, entities_per_document=entities_per_document
    )

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


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_not_starting_with_minus(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()
    entities_starting_with_minus = []
    for sentence in corpus.get_all_sentences():
        entities = sentence.get_spans("ner")
        for entity in entities:
            if str(entity.tokens[0].text).startswith("-"):
                entities_starting_with_minus.append(
                    " ".join([t.text for t in entity.tokens])
                )

    assert len(entities_starting_with_minus) == 0, "|".join(
        entities_starting_with_minus
    )


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_no_repeating_Bs(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()
    longest_repeat_tokens = []
    repeat_tokens = []
    for sentence in corpus.get_all_sentences():
        for token in sentence.tokens:
            if token.get_labels()[0].value.startswith("B") or token.get_labels()[
                0
            ].value.startswith("S"):
                repeat_tokens.append(token)
            else:
                if len(repeat_tokens) > len(longest_repeat_tokens):
                    longest_repeat_tokens = repeat_tokens
                repeat_tokens = []

    assert len(longest_repeat_tokens) < 4


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_no_long_entities(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()
    longest_entity = []
    for sentence in corpus.get_all_sentences():
        entities = sentence.get_spans("ner")
        for entity in entities:
            if len(entity.tokens) > len(longest_entity):
                longest_entity = [t.text for t in entity.tokens]

    assert len(longest_entity) < 10, " ".join(longest_entity)


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_no_unmatched_parentheses(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()
    unbalanced_entities = []
    for sentence in corpus.get_all_sentences():
        entities = sentence.get_spans("ner")
        for entity in entities:
            entity_text = "".join(t.text for t in entity.tokens)
            if not has_balanced_parantheses(entity_text):
                unbalanced_entities.append(entity_text)

    assert unbalanced_entities == []


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_not_too_many_entities(CorpusType: Type[ColumnCorpus]):
    corpus = CorpusType()
    n_entities_per_sentence = []
    for sentence in corpus.get_all_sentences():
        entities = sentence.get_spans("ner")
        n_entities_per_sentence.append(len(entities))
    avg_entities_per_sentence = sum(n_entities_per_sentence) / len(
        n_entities_per_sentence
    )

    assert avg_entities_per_sentence <= 5


@pytest.mark.parametrize("CorpusType", ALL_DATASETS)
@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_sanity_no_misaligned_entities(CorpusType: Type[HunerDataset]):
    dataset_name = CorpusType.__class__.__name__.lower()
    base_path = flair.cache_root / "datasets"
    data_folder = base_path / dataset_name

    from flair.tokenization import SciSpacyTokenizer
    tokenizer = SciSpacyTokenizer()

    corpus = CorpusType()
    internal = corpus.to_internal(data_folder)
    for doc_id, doc_text in internal.documents.items():
        misaligned_starts = []
        misaligned_ends = []

        token_starts = set()
        token_ends = set()
        for token, token_start in zip(*tokenizer.tokenize(doc_text)):
            token_starts.add(token_start)
            token_ends.add(token_start + len(token))

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


@pytest.mark.skip(msg="We skip this test because it's only relevant for development purposes")
def test_scispacy_tokenization():
    from flair.tokenization import SciSpacyTokenizer
    tokenizer = SciSpacyTokenizer()

    tokens = tokenizer.tokenize("HBeAg(+) patients")

    assert len(tokens) == 5
    assert tokens[0].text == "HBeAg"
    assert tokens[0].start_pos == 0
    assert tokens[1].text == "("
    assert tokens[1].start_pos == 5
    assert tokens[2].text == "+"
    assert tokens[2].start_pos == 6
    assert tokens[3].text == ")"
    assert tokens[3].start_pos == 7
    assert tokens[4].text == "patients"
    assert tokens[4].start_pos == 9

    tokens = tokenizer.tokenize("HBeAg(+)/HBsAg(+)")

    assert len(tokens) == 9

    assert tokens[0].text == "HBeAg"
    assert tokens[0].start_pos == 0
    assert tokens[1].text == "("
    assert tokens[1].start_pos == 5
    assert tokens[2].text == "+"
    assert tokens[2].start_pos == 6
    assert tokens[3].text == ")"
    assert tokens[3].start_pos == 7
    assert tokens[4].text == "/"
    assert tokens[4].start_pos == 8
    assert tokens[5].text == "HBsAg"
    assert tokens[5].start_pos == 9
    assert tokens[6].text == "("
    assert tokens[6].start_pos == 14
    assert tokens[7].text == "+"
    assert tokens[7].start_pos == 15
    assert tokens[8].text == ")"
    assert tokens[8].start_pos == 16

    tokens = tokenizer.tokenize("doxorubicin (DOX)-induced")

    assert len(tokens) == 5
    assert tokens[0].text == "doxorubicin"
    assert tokens[1].text == "("
    assert tokens[2].text == "DOX"
    assert tokens[3].text == ")"
    assert tokens[4].text == "-induced"

