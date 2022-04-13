import shutil

import pytest

import flair
import flair.datasets
from flair.data import MultiCorpus, Sentence
from flair.datasets import ColumnCorpus
from flair.datasets.sequence_labeling import (
    JsonlCorpus,
    JsonlDataset,
    MultiFileJsonlCorpus,
)


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

    assert [label.data_point.text for label in sent1.get_labels("ner")] == ["Larry Page", "Sergey Brin", "Google"]
    assert [label.value for label in sent1.get_labels("ner")] == ["PER", "PER", "ORG"]

    assert [token.get_label("upos").value for token in sent1.tokens] == [
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
        False,
        True,
    ]

    ner_spans1 = sent1.get_labels("ner")
    assert len(ner_spans1) == 3

    upos_spans1 = sent1.get_labels("upos")
    assert len(upos_spans1) == 8

    rels1 = sent1.get_labels("relation")
    assert len(rels1) == 2

    assert [token.idx for token in rels1[1].data_point.first] == [7]
    assert [token.idx for token in rels1[1].data_point.second] == [4, 5]

    sent3 = dataset[2]

    ner_labels3 = sent3.get_labels("ner")
    assert len(ner_labels3) == 3

    upos_labels3 = sent3.get_labels("upos")
    assert len(upos_labels3) == 11

    rels3 = sent3.get_labels("relation")
    assert len(rels3) == 1

    assert [token.idx for token in rels3[0].data_point.first] == [6]
    assert [token.idx for token in rels3[0].data_point.second] == [1, 2]


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

    assert [token.get_label("deprel").value for token in sent1.tokens] == [
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


def test_hipe_2022_corpus(tasks_base_path):
    """
    This test covers the complete v1.0 version of the HIPE 2022,
    including the version with document separator.
    """

    # We have manually checked, that these numbers are correct:
    hipe_stats = {
        "v1.0": {
            "ajmc": {
                "de": {"sample": {"sents": 119, "docs": 8, "labels": ["date", "loc", "pers", "scope", "work"]}},
                "en": {"sample": {"sents": 83, "docs": 5, "labels": ["date", "loc", "pers", "scope", "work"]}},
            },
            "hipe2020": {
                "de": {
                    "train": {
                        "sents": 3470 + 2,  # 2 sentences with missing EOS marker
                        "docs": 103,
                        "labels": ["loc", "org", "pers", "prod", "time"],
                    },
                    "dev": {"sents": 1202, "docs": 33, "labels": ["loc", "org", "pers", "prod", "time"]},
                },
                "en": {"dev": {"sents": 1045, "docs": 80, "labels": ["loc", "org", "pers", "prod", "time"]}},
                "fr": {
                    "train": {"sents": 5743, "docs": 158, "labels": ["loc", "org", "pers", "prod", "time", "comp"]},
                    "dev": {"sents": 1244, "docs": 43, "labels": ["loc", "org", "pers", "prod", "time"]},
                },
            },
            "letemps": {
                "fr": {
                    "train": {"sents": 14051, "docs": 414, "labels": ["loc", "org", "pers"]},
                    "dev": {"sents": 1341, "docs": 51, "labels": ["loc", "org", "pers"]},
                }
            },
            "newseye": {
                # +1 offset, because of missing EOS marker at EOD
                "de": {
                    "train": {"sents": 23646 + 1, "docs": 11, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev": {"sents": 1110 + 1, "docs": 12, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev2": {"sents": 1541 + 1, "docs": 12, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                },
                "fi": {
                    "train": {"sents": 1141 + 1, "docs": 24, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev": {"sents": 140 + 1, "docs": 24, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev2": {"sents": 104 + 1, "docs": 21, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                },
                "fr": {
                    "train": {"sents": 7106 + 1, "docs": 35, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev": {"sents": 662 + 1, "docs": 35, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev2": {"sents": 1016 + 1, "docs": 35, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                },
                "sv": {
                    "train": {"sents": 1063 + 1, "docs": 21, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev": {"sents": 126 + 1, "docs": 21, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                    "dev2": {"sents": 136 + 1, "docs": 21, "labels": ["HumanProd", "LOC", "ORG", "PER"]},
                },
            },
            "sonar": {
                "de": {
                    "dev": {
                        "sents": 1603 + 10,  # 10 sentences with missing EOS marker
                        "docs": 10,
                        "labels": ["LOC", "ORG", "PER"],
                    }
                }
            },
            "topres19th": {
                "en": {
                    "train": {"sents": 5874, "docs": 309, "labels": ["BUILDING", "LOC", "STREET"]},
                    "dev": {"sents": 646, "docs": 34, "labels": ["BUILDING", "LOC", "STREET"]},
                }
            },
        }
    }

    hipe_stats["v2.0"] = hipe_stats["v1.0"].copy()
    hipe_stats["v2.0"]["ajmc"] = {
        "de": {
            "train": {
                "sents": 1022 + 2,  # 2 sentences with missing EOS marker
                "docs": 76,
                "labels": ["date", "loc", "object", "pers", "scope", "work"],
            },
            "dev": {"sents": 192, "docs": 14, "labels": ["loc", "object", "pers", "scope", "work"]},
        },
        "en": {
            "train": {
                "sents": 1153 + 1,  # 1 sentence with missing EOS marker
                "docs": 60,
                "labels": ["date", "loc", "object", "pers", "scope", "work"],
            },
            "dev": {
                "sents": 251 + 1,  # 1 sentence with missing EOS marker
                "docs": 14,
                "labels": ["date", "loc", "pers", "scope", "work"],
            },
        },
        "fr": {
            "train": {
                "sents": 893 + 1,  # 1 sentence with missing EOS marker
                "docs": 72,
                "labels": ["date", "loc", "object", "pers", "scope", "work"],
            },
            "dev": {
                "sents": 201 + 1,  # 1 sentence with missing EOS marker
                "docs": 17,
                "labels": ["pers", "scope", "work"],
            },
        },
    }
    hipe_stats["v2.0"]["newseye"] = {
        "de": {
            "train": {"sents": 20839 + 1, "docs": 7, "labels": ["HumanProd", "LOC", "ORG", "PER"]}  # missing EOD marker
        }
    }
    hipe_stats["v2.0"]["sonar"] = {
        "de": {
            "dev": {
                "sents": 816 + 10,  # 9 sentences with missing EOS marker + missing EOD
                "docs": 10,
                "labels": ["LOC", "ORG", "PER"],
            }
        }
    }

    def test_hipe_2022(dataset_version="v1.0", add_document_separator=True):
        for dataset_name, languages in hipe_stats[dataset_version].items():
            for language in languages:
                splits = languages[language]

                corpus = flair.datasets.NER_HIPE_2022(
                    version=dataset_version,
                    dataset_name=dataset_name,
                    language=language,
                    dev_split_name="dev",
                    add_document_separator=add_document_separator,
                )

                for split_name, stats in splits.items():
                    split_description = f"{dataset_name}/{language}@{split_name}"

                    current_sents = stats["sents"]
                    current_docs = stats["docs"]
                    current_labels = set(stats["labels"] + ["<unk>"])

                    total_sentences = current_sents + current_docs if add_document_separator else stats["sents"]

                    if split_name == "train":
                        assert (
                            len(corpus.train) == total_sentences
                        ), f"Sentence count mismatch for {split_description}: {len(corpus.train)} vs. {total_sentences}"

                        gold_labels = set(corpus.make_label_dictionary(label_type="ner").get_items())

                        assert (
                            current_labels == gold_labels
                        ), f"Label mismatch for {split_description}: {current_labels} vs. {gold_labels}"

                    elif split_name in ["dev", "sample"]:
                        assert (
                            len(corpus.dev) == total_sentences
                        ), f"Sentence count mismatch for {split_description}: {len(corpus.dev)} vs. {total_sentences}"

                        corpus._train = corpus._dev
                        gold_labels = set(corpus.make_label_dictionary(label_type="ner").get_items())

                        assert (
                            current_labels == gold_labels
                        ), f"Label mismatch for {split_description}: {current_labels} vs. {gold_labels}"
                    elif split_name == "dev2":
                        corpus = flair.datasets.NER_HIPE_2022(
                            version=dataset_version,
                            dataset_name=dataset_name,
                            language=language,
                            dev_split_name="dev2",
                            add_document_separator=add_document_separator,
                        )

                        corpus._train = corpus._dev
                        gold_labels = set(corpus.make_label_dictionary(label_type="ner").get_items())

                        assert (
                            len(corpus.dev) == total_sentences
                        ), f"Sentence count mismatch for {split_description}: {len(corpus.dev)} vs. {total_sentences}"

                        assert (
                            current_labels == gold_labels
                        ), f"Label mismatch for {split_description}: {current_labels} vs. {gold_labels}"

    test_hipe_2022(dataset_version="v1.0", add_document_separator=True)
    test_hipe_2022(dataset_version="v1.0", add_document_separator=False)
    test_hipe_2022(dataset_version="v2.0", add_document_separator=True)
    test_hipe_2022(dataset_version="v2.0", add_document_separator=False)


def test_multi_file_jsonl_corpus_should_use_label_type(tasks_base_path):
    corpus = MultiFileJsonlCorpus(
        train_files=[tasks_base_path / "jsonl/train.jsonl"],
        dev_files=[tasks_base_path / "jsonl/testa.jsonl"],
        test_files=[tasks_base_path / "jsonl/testb.jsonl"],
        label_type="pos",
    )

    for sentence in corpus.get_all_sentences():
        assert sentence.has_label("pos")
        assert not sentence.has_label("ner")


def test_jsonl_corpus_should_use_label_type(tasks_base_path):
    corpus = JsonlCorpus(tasks_base_path / "jsonl", label_type="pos")

    for sentence in corpus.get_all_sentences():
        assert sentence.has_label("pos")
        assert not sentence.has_label("ner")


def test_jsonl_dataset_should_use_label_type(tasks_base_path):
    """
    Tests whether the dataset respects the label_type parameter
    """
    dataset = JsonlDataset(tasks_base_path / "jsonl/train.jsonl", label_type="pos")  # use other type

    for sentence in dataset.sentences:
        assert sentence.has_label("pos")
        assert not sentence.has_label("ner")


def test_reading_jsonl_dataset_should_be_successful(tasks_base_path):
    """
    Tests reading a JsonlDataset containing multiple tagged entries
    """
    dataset = JsonlDataset(tasks_base_path / "jsonl/train.jsonl")

    assert len(dataset.sentences) == 5
    assert dataset.sentences[0].get_token(3).get_label("ner").value == "B-LOC"
    assert dataset.sentences[0].get_token(4).get_label("ner").value == "I-LOC"


def test_simple_folder_jsonl_corpus_should_load(tasks_base_path):
    corpus = JsonlCorpus(tasks_base_path / "jsonl")
    assert len(corpus.get_all_sentences()) == 11


TRAIN_FILE = "tests/resources/tasks/jsonl/train.jsonl"
TESTA_FILE = "tests/resources/tasks/jsonl/testa.jsonl"
TESTB_FILE = "tests/resources/tasks/jsonl/testa.jsonl"


@pytest.mark.parametrize(
    "train_files,dev_files,test_files,expected_size",
    [
        (
            [TRAIN_FILE],
            [TESTA_FILE],
            [TESTB_FILE],
            11,
        ),
        (
            [TRAIN_FILE],
            [],
            [TESTB_FILE],
            8,
        ),
        (
            [TRAIN_FILE],
            [],
            None,
            5,
        ),
        (
            None,
            [TESTA_FILE],
            None,
            3,
        ),
        (
            [TRAIN_FILE, TESTA_FILE],
            [TESTA_FILE],
            [TESTB_FILE],
            14,
        ),
    ],
)
def test_corpus_with_single_files_should_load(train_files, dev_files, test_files, expected_size):
    corpus = MultiFileJsonlCorpus(train_files, dev_files, test_files)
    assert len(corpus.get_all_sentences()) == expected_size


def test_empty_corpus_should_raise_error():
    with pytest.raises(RuntimeError) as err:
        MultiFileJsonlCorpus(None, None, None)

    assert str(err.value) == "No data provided when initializing corpus object."
