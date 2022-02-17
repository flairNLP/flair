import shutil

import flair
import flair.datasets
from flair.data import MultiCorpus, Sentence
from flair.datasets import ColumnCorpus


def test_load_imdb_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(
        tasks_base_path / "imdb",
        memory_mode="full",
    )

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 5


def test_load_imdb_data_streaming(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(
        tasks_base_path / "imdb",
        memory_mode="disk",
    )

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 5


def test_load_imdb_data_max_tokens(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", memory_mode="full", truncate_to_max_tokens=3)

    assert len(corpus.train[0]) <= 3
    assert len(corpus.dev[0]) <= 3
    assert len(corpus.test[0]) <= 3


def test_load_imdb_data_streaming_max_tokens(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", memory_mode="full", truncate_to_max_tokens=3)

    assert len(corpus.train[0]) <= 3
    assert len(corpus.dev[0]) <= 3
    assert len(corpus.test[0]) <= 3


def test_load_ag_news_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "ag_news")

    assert len(corpus.train) == 10
    assert len(corpus.dev) == 10
    assert len(corpus.test) == 10


def test_load_sequence_labeling_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(tasks_base_path / "fashion", column_format={0: "text", 2: "ner"})

    assert len(corpus.train) == 6
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_sequence_labeling_whitespace_after(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(
        tasks_base_path / "column_with_whitespaces",
        column_format={0: "text", 1: "ner", 2: "space-after"},
    )

    assert len(corpus.train) == 1
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1

    assert corpus.train[0].to_tokenized_string() == "It is a German - owned firm ."
    assert corpus.train[0].to_plain_string() == "It is a German-owned firm."
    for token in corpus.train[0]:
        assert token.start_pos is not None
        assert token.end_pos is not None


def test_load_column_corpus_options(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(
        tasks_base_path / "column_corpus_options",
        column_format={0: "text", 1: "ner"},
        column_delimiter="\t",
        skip_first_line=True,
    )

    assert len(corpus.train) == 1
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1

    assert corpus.train[0].to_tokenized_string() == "This is New Berlin"


def test_load_germeval_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(tasks_base_path / "germeval_14", column_format={0: "text", 2: "ner"})

    assert len(corpus.train) == 2
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_ud_english_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.UD_ENGLISH(tasks_base_path)

    assert len(corpus.train) == 6
    assert len(corpus.test) == 4
    assert len(corpus.dev) == 2


def test_load_no_dev_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(tasks_base_path / "fashion_nodev", column_format={0: "text", 2: "ner"})

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_no_dev_data_explicit(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(
        tasks_base_path / "fashion_nodev",
        column_format={0: "text", 2: "ner"},
        train_file="train.tsv",
        test_file="test.tsv",
    )

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_multi_corpus(tasks_base_path):
    corpus_1 = flair.datasets.ColumnCorpus(tasks_base_path / "germeval_14", column_format={0: "text", 2: "ner"})

    corpus_2 = flair.datasets.ColumnCorpus(tasks_base_path / "fashion", column_format={0: "text", 2: "ner"})
    # get two corpora as one
    corpus = MultiCorpus([corpus_1, corpus_2])

    assert len(corpus.train) == 8
    assert len(corpus.dev) == 2
    assert len(corpus.test) == 2


def test_download_load_data(tasks_base_path):
    # get training, test and dev data for full English UD corpus from web
    corpus = flair.datasets.UD_ENGLISH()

    assert len(corpus.train) == 12543
    assert len(corpus.dev) == 2001
    assert len(corpus.test) == 2077

    # clean up data directory
    shutil.rmtree(flair.cache_root / "datasets" / "ud_english")


def _assert_conllu_dataset(dataset):
    sent1 = dataset[0]

    assert [entity.span.text for entity in sent1.get_labels("ner")] == ["Larry Page", "Sergey Brin", "Google"]
    assert [entity.value for entity in sent1.get_labels("ner")] == ["PER", "PER", "ORG"]

    assert [token.get_tag("upos").value for token in sent1.tokens] == [
        "PROPN",
        "PROPN",
        "CCONJ",
        "PROPN",
        "PROPN",
        "VERB",
        "PROPN",
        "PUNCT",
    ]

    assert [token.whitespace_after for token in sent1.tokens] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    ner_spans1 = sent1.get_labels("ner")
    assert len(ner_spans1) == 3

    upos_spans1 = sent1.get_labels("upos")
    assert len(upos_spans1) == 8

    rels1 = sent1.get_labels("relation")
    assert len(rels1) == 2

    assert [token.idx for token in rels1[1].head] == [7]
    assert [token.idx for token in rels1[1].tail] == [4, 5]

    sent3 = dataset[2]

    ner_labels3 = sent3.get_labels("ner")
    assert len(ner_labels3) == 3

    upos_labels3 = sent3.get_labels("upos")
    assert len(upos_labels3) == 11

    rels3 = sent3.get_labels("relation")
    assert len(rels3) == 1

    assert [token.idx for token in rels3[0].head] == [6]
    assert [token.idx for token in rels3[0].tail] == [1, 2]


def test_load_conllu_corpus(tasks_base_path):
    corpus = ColumnCorpus(
        tasks_base_path / "conllu",
        train_file="train.conllu",
        dev_file="train.conllu",
        test_file="train.conllu",
        in_memory=False,
        column_format={1: "text", 2: "upos", 3: "ner", 4: "feats"},
    )

    assert len(corpus.train) == 4
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 4

    _assert_conllu_dataset(corpus.train)


def test_load_conllu_corpus_in_memory(tasks_base_path):
    corpus = ColumnCorpus(
        tasks_base_path / "conllu",
        train_file="train.conllu",
        dev_file="train.conllu",
        test_file="train.conllu",
        column_format={1: "text", 2: "upos", 3: "ner", 4: "feats"},
        in_memory=True,
    )

    assert len(corpus.train) == 4
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 4

    _assert_conllu_dataset(corpus.train)


def test_load_conllu_plus_corpus(tasks_base_path):
    corpus = ColumnCorpus(
        tasks_base_path / "conllu",
        train_file="train.conllup",
        dev_file="train.conllup",
        test_file="train.conllup",
        column_format={1: "text", 2: "upos", 3: "ner", 4: "feats"},
        in_memory=False,
    )

    assert len(corpus.train) == 4
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 4

    _assert_conllu_dataset(corpus.train)


def test_load_conllu_corpus_plus_in_memory(tasks_base_path):
    corpus = ColumnCorpus(
        tasks_base_path / "conllu",
        train_file="train.conllup",
        dev_file="train.conllup",
        test_file="train.conllup",
        column_format={1: "text", 2: "upos", 3: "ner", 4: "feats"},
        in_memory=True,
    )

    assert len(corpus.train) == 4
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 4

    _assert_conllu_dataset(corpus.train)


def _assert_universal_dependencies_conllu_dataset(dataset):
    sent1: Sentence = dataset[0]

    assert [token.whitespace_after for token in sent1.tokens] == [
        True,
        True,
        True,
        True,
        False,
        True,
    ]

    assert len(sent1.get_labels("Number")) == 4
    assert sent1[1].get_labels("Number")[0].value == "Plur"
    assert sent1[1].get_labels("Person")[0].value == "3"
    assert sent1[1].get_labels("Tense")[0].value == "Pres"

    # assert [token.get_tag("head").value for token in sent1.tokens] == [
    #     "2",
    #     "0",
    #     "4",
    #     "2",
    #     "2",
    #     "2",
    # ]

    assert [token.get_tag("deprel").value for token in sent1.tokens] == [
        "nsubj",
        "root",
        "cc",
        "conj",
        "obj",
        "punct",
    ]


def test_load_universal_dependencies_conllu_corpus(tasks_base_path):
    """
    This test only covers basic universal dependencies datasets.
    For example, multi-word tokens or the "deps" column sentence annotations
    are not supported yet.
    """

    # Here, we use the default token annotation fields.
    corpus = ColumnCorpus(
        tasks_base_path / "conllu",
        train_file="universal_dependencies.conllu",
        dev_file="universal_dependencies.conllu",
        test_file="universal_dependencies.conllu",
        column_format={
            1: "text",
            2: "lemma",
            3: "upos",
            4: "pos",
            5: "feats",
            6: "head",
            7: "deprel",
            8: "deps",
            9: "misc",
        },
    )

    assert len(corpus.train) == 1
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1

    _assert_universal_dependencies_conllu_dataset(corpus.train)
