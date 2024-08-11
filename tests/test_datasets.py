import copy
import shutil

import pytest

import flair
import flair.datasets
from flair.data import MultiCorpus, Sentence
from flair.datasets import ColumnCorpus
from flair.datasets.sequence_labeling import (
    ONTONOTES,
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
        assert token.start_position is not None
        assert token.end_position is not None


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


def test_load_span_data(tasks_base_path):
    # load column dataset with one entry
    dataset = flair.datasets.ColumnDataset(
        tasks_base_path / "span_labels" / "span_first.txt",
        column_name_map={0: "text", 1: "ner"},
    )

    assert len(dataset) == 1
    assert dataset[0][2].text == "RAB"
    assert dataset[0][2].get_label("ner").value == "PARTA"

    # load column dataset with two entries
    dataset = flair.datasets.ColumnDataset(
        tasks_base_path / "span_labels" / "span_second.txt",
        column_name_map={0: "text", 1: "ner"},
    )

    assert len(dataset) == 2
    assert dataset[1][2].text == "RAB"
    assert dataset[1][2].get_label("ner").value == "PARTA"

    # load column dataset with three entries
    dataset = flair.datasets.ColumnDataset(
        tasks_base_path / "span_labels" / "span_third.txt",
        column_name_map={0: "text", 1: "ner"},
    )

    assert len(dataset) == 3
    assert dataset[2][2].text == "RAB"
    assert dataset[2][2].get_label("ner").value == "PARTA"


def test_load_germeval_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.ColumnCorpus(tasks_base_path / "ner_german_germeval", column_format={0: "text", 2: "ner"})

    assert len(corpus.train) == 2
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_ud_english_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.UD_ENGLISH(tasks_base_path)

    assert len(corpus.train) == 6
    assert len(corpus.test) == 4
    assert len(corpus.dev) == 2

    # check if Token labels are correct
    sentence = corpus.train[0]
    assert sentence[0].text == "From"
    assert sentence[0].get_label("upos").value == "ADP"
    assert sentence[1].text == "the"
    assert sentence[1].get_label("upos").value == "DET"


def test_load_up_english_data(tasks_base_path):
    # get training, test and dev data
    corpus = flair.datasets.UP_ENGLISH(tasks_base_path)

    assert len(corpus.train) == 4
    assert len(corpus.test) == 2
    assert len(corpus.dev) == 2

    # check if Token labels for frames are correct
    sentence = corpus.dev[0]
    assert sentence[2].text == "AP"
    assert sentence[2].get_label("frame", zero_tag_value="no_label").value == "no_label"
    assert sentence[3].text == "comes"
    assert sentence[3].get_label("frame").value == "come.03"


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
    corpus_1 = flair.datasets.ColumnCorpus(tasks_base_path / "ner_german_germeval", column_format={0: "text", 2: "ner"})

    corpus_2 = flair.datasets.ColumnCorpus(tasks_base_path / "fashion", column_format={0: "text", 2: "ner"})
    # get two corpora as one
    corpus = MultiCorpus([corpus_1, corpus_2])

    assert len(corpus.train) == 8
    assert len(corpus.dev) == 2
    assert len(corpus.test) == 2


def test_download_load_data(tasks_base_path):
    # get training, test and dev data for full English UD corpus from web
    corpus = flair.datasets.UD_ENGLISH()

    assert len(corpus.train) == 12544
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
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
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
        1,
        1,
        1,
        1,
        0,
        0,
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
    # This test only covers basic universal dependencies datasets.
    # For example, multi-word tokens or the "deps" column sentence annotations are not supported yet.

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


@pytest.mark.skip()
def test_hipe_2022_corpus(tasks_base_path):
    # This test covers the complete HIPE 2022 dataset.
    # https://github.com/hipe-eval/HIPE-2022-data
    # Includes variant with document separator, and all versions of the dataset.

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

    hipe_stats["v2.0"] = copy.deepcopy(hipe_stats["v1.0"])
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
    hipe_stats["v2.0"]["newseye"]["de"] = {
        "train": {"sents": 20839 + 1, "docs": 7, "labels": ["HumanProd", "LOC", "ORG", "PER"]}  # missing EOD marker
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
    hipe_stats["v2.1"] = copy.deepcopy(hipe_stats["v2.0"])
    hipe_stats["v2.1"]["hipe2020"]["fr"]["train"] = {
        "sents": 5743,
        "docs": 158,
        "labels": ["loc", "org", "pers", "prod", "time"],
    }

    # Test data for v2.1 release
    hipe_stats["v2.1"]["ajmc"]["de"]["test"] = {
        "sents": 224,
        "docs": 16,
        "labels": ["loc", "object", "pers", "scope", "work"],
    }
    hipe_stats["v2.1"]["ajmc"]["en"]["test"] = {
        "sents": 238,
        "docs": 13,
        "labels": ["date", "loc", "pers", "scope", "work"],
    }
    hipe_stats["v2.1"]["ajmc"]["fr"]["test"] = {
        "sents": 188 + 1,  # 1 sentence with missing EOS marker
        "docs": 15,
        "labels": ["date", "loc", "pers", "scope", "work"],
    }
    hipe_stats["v2.1"]["hipe2020"]["de"]["test"] = {
        "sents": 1215 + 2,  # 2 sentences with missing EOS marker
        "docs": 49,
        "labels": ["loc", "org", "pers", "prod", "time"],
    }
    hipe_stats["v2.1"]["hipe2020"]["en"]["test"] = {
        "sents": 553,
        "docs": 46,
        "labels": ["loc", "org", "pers", "prod", "time"],
    }
    hipe_stats["v2.1"]["hipe2020"]["fr"]["test"] = {
        "sents": 1462,
        "docs": 43,
        "labels": ["loc", "org", "pers", "prod", "time"],
    }
    hipe_stats["v2.1"]["letemps"]["fr"]["test"] = {"sents": 2381, "docs": 51, "labels": ["loc", "org", "pers"]}
    hipe_stats["v2.1"]["newseye"]["de"]["test"] = {
        "sents": 3336 + 1,  # 1 missing EOD marker
        "docs": 13,
        "labels": ["HumanProd", "LOC", "ORG", "PER"],
    }
    hipe_stats["v2.1"]["newseye"]["fi"]["test"] = {
        "sents": 390 + 1,  # 1 missing EOD marker
        "docs": 24,
        "labels": ["HumanProd", "LOC", "ORG", "PER"],
    }
    hipe_stats["v2.1"]["newseye"]["fr"]["test"] = {
        "sents": 2534 + 1,  # 1 missing EOD marker
        "docs": 35,
        "labels": ["HumanProd", "LOC", "ORG", "PER"],
    }
    hipe_stats["v2.1"]["newseye"]["sv"]["test"] = {
        "sents": 342 + 1,  # 1 missing EOD marker
        "docs": 21,
        "labels": ["HumanProd", "LOC", "ORG", "PER"],
    }
    hipe_stats["v2.1"]["sonar"]["de"]["test"] = {
        "sents": 807 + 8 + 1,  # 8 missing EOS marker + missing EOD
        "docs": 10,
        "labels": ["LOC", "ORG", "PER"],
    }
    hipe_stats["v2.1"]["topres19th"]["en"]["test"] = {
        "sents": 2001,
        "docs": 112,
        "labels": ["BUILDING", "LOC", "STREET"],
    }

    def test_hipe_2022(dataset_version="v2.1", add_document_separator=True):
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
                    split_description = f"{dataset_name}@{dataset_version}/{language}#{split_name}"

                    current_sents = stats["sents"]
                    current_docs = stats["docs"]
                    current_labels = set(stats["labels"])

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
    test_hipe_2022(dataset_version="v2.1", add_document_separator=True)
    test_hipe_2022(dataset_version="v2.1", add_document_separator=False)


@pytest.mark.skip()
def test_icdar_europeana_corpus(tasks_base_path):
    # This test covers the complete ICDAR Europeana corpus:
    # https://github.com/stefan-it/historic-domain-adaptation-icdar

    gold_stats = {"fr": {"train": 7936, "dev": 992, "test": 992}, "nl": {"train": 5777, "dev": 722, "test": 723}}

    def check_number_sentences(reference: int, actual: int, split_name: str):
        assert actual == reference, f"Mismatch in number of sentences for {split_name} split"

    for language in ["fr", "nl"]:
        corpus = flair.datasets.NER_ICDAR_EUROPEANA(language=language)

        check_number_sentences(len(corpus.train), gold_stats[language]["train"], "train")
        check_number_sentences(len(corpus.dev), gold_stats[language]["dev"], "dev")
        check_number_sentences(len(corpus.test), gold_stats[language]["test"], "test")


@pytest.mark.skip()
def test_masakhane_corpus(tasks_base_path):
    # This test covers the complete MasakhaNER dataset, including support for v1 and v2.
    supported_versions = ["v1", "v2"]

    supported_languages = {
        "v1": ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"],
        "v2": [
            "bam",
            "bbj",
            "ewe",
            "fon",
            "hau",
            "ibo",
            "kin",
            "lug",
            "mos",
            "pcm",
            "nya",
            "sna",
            "swa",
            "tsn",
            "twi",
            "wol",
            "xho",
            "yor",
            "zul",
        ],
    }

    masakhane_stats = {
        "v1": {
            "amh": {"train": 1750, "dev": 250, "test": 500},
            "hau": {"train": 1912, "dev": 276, "test": 552},
            "ibo": {"train": 2235, "dev": 320, "test": 638},
            "kin": {
                "train": 2116,
                "dev": 302,
                "test": 605,
            },
            "lug": {"train": 1428, "dev": 200, "test": 407},
            "luo": {"train": 644, "dev": 92, "test": 186},
            "pcm": {"train": 2124, "dev": 306, "test": 600},
            "swa": {"train": 2109, "dev": 300, "test": 604},
            "wol": {"train": 1871, "dev": 267, "test": 539},
            "yor": {"train": 2171, "dev": 305, "test": 645},
        },
        "v2": {
            "bam": {"train": 4462, "dev": 638, "test": 1274},
            "bbj": {"train": 3384, "dev": 483, "test": 966},
            "ewe": {"train": 3505, "dev": 501, "test": 1001},
            "fon": {"train": 4343, "dev": 623, "test": 1228},
            "hau": {"train": 5716, "dev": 816, "test": 1633},
            "ibo": {"train": 7634, "dev": 1090, "test": 2181},
            "kin": {"train": 7825, "dev": 1118, "test": 2235},
            "lug": {"train": 4942, "dev": 706, "test": 1412},
            "mos": {"train": 4532, "dev": 648, "test": 1294},
            "pcm": {"train": 5646, "dev": 806, "test": 1613},
            "nya": {"train": 6250, "dev": 893, "test": 1785},
            "sna": {"train": 6207, "dev": 887, "test": 1773},
            "swa": {"train": 6593, "dev": 942, "test": 1883},
            "tsn": {"train": 3489, "dev": 499, "test": 996},
            "twi": {"train": 4240, "dev": 605, "test": 1211},
            "wol": {"train": 4593, "dev": 656, "test": 1312},
            "xho": {"train": 5718, "dev": 817, "test": 1633},
            "yor": {"train": 6876, "dev": 983, "test": 1964},
            "zul": {"train": 5848, "dev": 836, "test": 1670},
        },
    }

    def check_number_sentences(reference: int, actual: int, split_name: str, language: str, version: str):
        assert actual == reference, f"Mismatch in number of sentences for {language}@{version}/{split_name}"

    for version in supported_versions:
        for language in supported_languages[version]:
            corpus = flair.datasets.NER_MASAKHANE(languages=language, version=version)

            gold_stats = masakhane_stats[version][language]

            check_number_sentences(len(corpus.train), gold_stats["train"], "train", language, version)
            check_number_sentences(len(corpus.dev), gold_stats["dev"], "dev", language, version)
            check_number_sentences(len(corpus.test), gold_stats["test"], "test", language, version)


@pytest.mark.skip()
def test_nermud_corpus(tasks_base_path):
    # This test covers the NERMuD dataset. Official stats can be found here:
    # https://github.com/dhfbk/KIND/tree/main/evalita-2023
    gold_stats = {
        "WN": {"train": 10912, "dev": 2594},
        "FIC": {"train": 11423, "dev": 1051},
        "ADG": {"train": 5147, "dev": 1122},
    }

    def check_number_sentences(reference: int, actual: int, split_name: str):
        assert actual == reference, f"Mismatch in number of sentences for {split_name} split"

    for domain, stats in gold_stats.items():
        corpus = flair.datasets.NER_NERMUD(domains=domain)
        check_number_sentences(len(corpus.train), stats["train"], "train")
        check_number_sentences(len(corpus.dev), stats["dev"], "dev")


@pytest.mark.skip()
def test_german_ler_corpus(tasks_base_path):
    corpus = flair.datasets.NER_GERMAN_LEGAL()

    # Number of instances per dataset split are taken from https://huggingface.co/datasets/elenanereiss/german-ler
    assert len(corpus.train) == 53384, "Mismatch in number of sentences for train split"
    assert len(corpus.dev) == 6666, "Mismatch in number of sentences for dev split"
    assert len(corpus.test) == 6673, "Mismatch in number of sentences for test split"


@pytest.mark.skip()
def test_masakha_pos_corpus(tasks_base_path):
    # This test covers the complete MasakhaPOS dataset.
    supported_versions = ["v1"]

    supported_languages = {
        "v1": [
            "bam",
            "bbj",
            "ewe",
            "fon",
            "hau",
            "ibo",
            "kin",
            "lug",
            "luo",
            "mos",
            "pcm",
            "nya",
            "sna",
            "swa",
            "tsn",
            "twi",
            "wol",
            "xho",
            "yor",
            "zul",
        ],
    }

    masakha_pos_stats = {
        "v1": {
            "bam": {"train": 775, "dev": 154, "test": 619},
            "bbj": {"train": 750, "dev": 149, "test": 599},
            "ewe": {"train": 728, "dev": 145, "test": 582},
            "fon": {"train": 810, "dev": 161, "test": 646},
            "hau": {"train": 753, "dev": 150, "test": 601},
            "ibo": {"train": 803, "dev": 160, "test": 642},
            "kin": {"train": 757, "dev": 151, "test": 604},
            "lug": {"train": 733, "dev": 146, "test": 586},
            "luo": {"train": 758, "dev": 151, "test": 606},
            "mos": {"train": 757, "dev": 151, "test": 604},
            "pcm": {"train": 752, "dev": 150, "test": 600},
            "nya": {"train": 728, "dev": 145, "test": 582},
            "sna": {"train": 747, "dev": 149, "test": 596},
            "swa": {"train": 693, "dev": 138, "test": 553},
            "tsn": {"train": 754, "dev": 150, "test": 602},
            "twi": {"train": 785, "dev": 157, "test": 628},
            "wol": {"train": 782, "dev": 156, "test": 625},
            "xho": {"train": 752, "dev": 150, "test": 601},
            "yor": {"train": 893, "dev": 178, "test": 713},
            "zul": {"train": 753, "dev": 150, "test": 601},
        },
    }

    def check_number_sentences(reference: int, actual: int, split_name: str, language: str, version: str):
        assert actual == reference, f"Mismatch in number of sentences for {language}@{version}/{split_name}"

    for version in supported_versions:
        for language in supported_languages[version]:
            corpus = flair.datasets.MASAKHA_POS(languages=language, version=version)

            gold_stats = masakha_pos_stats[version][language]

            check_number_sentences(len(corpus.train), gold_stats["train"], "train", language, version)
            check_number_sentences(len(corpus.dev), gold_stats["dev"], "dev", language, version)
            check_number_sentences(len(corpus.test), gold_stats["test"], "test", language, version)


@pytest.mark.skip()
def test_german_mobie(tasks_base_path):
    corpus = flair.datasets.NER_GERMAN_MOBIE()

    # See MobIE paper (https://aclanthology.org/2021.konvens-1.22/), table 2
    ref_sentences = 7_077
    ref_tokens = 90_971

    actual_sentences = sum(
        [1 for sentence in corpus.train + corpus.dev + corpus.test if sentence[0].text != "-DOCSTART-"]
    )
    actual_tokens = sum(
        [len(sentence) for sentence in corpus.train + corpus.dev + corpus.test if sentence[0].text != "-DOCSTART-"]
    )

    assert ref_sentences == actual_sentences, (
        f"Number of parsed sentences ({actual_sentences}) does not match with "
        f"reported number of sentences ({ref_sentences})!"
    )
    assert (
        ref_tokens == actual_tokens
    ), f"Number of parsed tokens ({actual_tokens}) does not match with reported number of tokens ({ref_tokens})!"


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
    """Tests whether the dataset respects the label_type parameter."""
    dataset = JsonlDataset(tasks_base_path / "jsonl" / "train.jsonl", label_type="pos")  # use other type

    for sentence in dataset.sentences:
        assert sentence.has_label("pos")
        assert not sentence.has_label("ner")


def test_reading_jsonl_dataset_should_be_successful(tasks_base_path):
    """Tests reading a JsonlDataset containing multiple tagged entries."""
    dataset = JsonlDataset(tasks_base_path / "jsonl" / "train.jsonl")

    assert len(dataset.sentences) == 5
    assert len(dataset.sentences[0].get_labels("ner")) == 1
    assert dataset.sentences[0][2:4].get_label("ner").value == "LOC"


def test_simple_folder_jsonl_corpus_should_load(tasks_base_path):
    corpus = JsonlCorpus(tasks_base_path / "jsonl")
    assert len(corpus.get_all_sentences()) == 11


def test_jsonl_corpus_loads_spans(tasks_base_path):
    corpus = JsonlCorpus(tasks_base_path / "jsonl")
    assert corpus.train is not None
    example = corpus.train[0]
    assert len(example.get_spans("ner")) > 0


def test_jsonl_corpus_loads_metadata(tasks_base_path):
    """Tests reading a JsonlDataset containing metadata."""
    dataset = JsonlDataset(tasks_base_path / "jsonl" / "testa.jsonl")

    assert len(dataset.sentences) == 3
    assert dataset.sentences[0].get_metadata("from") == 123
    assert dataset.sentences[1].get_metadata("from") == 124
    assert dataset.sentences[2].get_metadata("from") == 125


@pytest.mark.skip()
def test_ontonotes_download():
    from urllib.parse import urlparse

    res = urlparse(ONTONOTES.archive_url)
    assert all([res.scheme, res.netloc])


@pytest.mark.skip()
def test_ontonotes_extraction(tasks_base_path):
    import os
    import tempfile

    from flair.file_utils import unpack_file

    ontonotes_path = tasks_base_path / "ontonotes"
    with tempfile.TemporaryDirectory() as tmp_dir:
        unpack_file(ontonotes_path / "tiny-conll-2012.zip", tmp_dir, "zip", True)
        assert "conll-2012" in os.listdir(tmp_dir)

        corpus = ONTONOTES(base_path=tmp_dir)
        label_dictionary = corpus.make_label_dictionary("ner")

        assert len(label_dictionary) == 14
        assert label_dictionary.span_labels

        domain_specific_corpus = ONTONOTES(base_path=tmp_dir, domain=["bc"])

        assert len(corpus.train) > len(domain_specific_corpus.train)


TRAIN_FILE = "tests/resources/tasks/jsonl/train.jsonl"
TESTA_FILE = "tests/resources/tasks/jsonl/testa.jsonl"
TESTB_FILE = "tests/resources/tasks/jsonl/testa.jsonl"


@pytest.mark.parametrize(
    ("train_files", "dev_files", "test_files", "expected_size"),
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
