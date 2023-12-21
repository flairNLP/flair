import os

import pytest

import flair
from flair.data import Corpus, Dictionary, Label, Sentence
from flair.datasets import ColumnCorpus, FlairDatapointDataset, SentenceDataset


def test_dictionary_get_items_with_unk():
    dictionary: Dictionary = Dictionary(add_unk=True)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert len(items) == 4
    assert items[0] == "<unk>"
    assert items[1] == "class_1"
    assert items[2] == "class_2"
    assert items[3] == "class_3"


def test_dictionary_get_items_without_unk():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert len(items) == 3
    assert items[0] == "class_1"
    assert items[1] == "class_2"
    assert items[2] == "class_3"


def test_dictionary_get_idx_for_item():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    idx = dictionary.get_idx_for_item("class_2")

    assert idx == 1


def test_dictionary_get_item_for_index():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    item = dictionary.get_item_for_index(0)

    assert item == "class_1"


def test_dictionary_save_and_load():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    file_path = "dictionary.txt"

    dictionary.save(file_path)
    loaded_dictionary = dictionary.load_from_file(file_path)

    assert len(dictionary) == len(loaded_dictionary)
    assert len(dictionary.get_items()) == len(loaded_dictionary.get_items())

    # clean up file
    os.remove(file_path)


def test_deprecated_sentence_dataset():
    with pytest.warns(DeprecationWarning):  # test to make sure the warning comes, but class works
        SentenceDataset([Sentence("Short sentences are short")])


def test_tagged_corpus_get_all_sentences():
    train_sentence = Sentence("I'm used in training.")
    dev_sentence = Sentence("I'm a dev sentence.")
    test_sentence = Sentence("I will be only used for testing.")

    corpus: Corpus = Corpus(
        FlairDatapointDataset([train_sentence]),
        FlairDatapointDataset([dev_sentence]),
        FlairDatapointDataset([test_sentence]),
    )

    all_sentences = corpus.get_all_sentences()

    assert len(all_sentences) == 3


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence("used in training. training is cool.")

    corpus: Corpus = Corpus(FlairDatapointDataset([train_sentence]), sample_missing_splits=False)

    vocab = corpus.make_vocab_dictionary(max_tokens=2, min_freq=-1)

    assert len(vocab) == 3
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=-1)

    assert len(vocab) == 7

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=2)

    assert len(vocab) == 3
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()


def test_label_set_confidence():
    label = Label(data_point=None, value="class_1", score=3.2)

    assert label.score == 3.2
    assert label.value == "class_1"

    label._score = 0.2

    assert label.score == 0.2


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence("sentence 1").add_label("label", "class_1")

    sentence_2 = Sentence("sentence 2").add_label("label", "class_2")

    sentence_3 = Sentence("sentence 3").add_label("label", "class_1")

    corpus: Corpus = Corpus(
        FlairDatapointDataset([sentence_1, sentence_2, sentence_3]),
        FlairDatapointDataset([]),
        FlairDatapointDataset([]),
    )

    label_dict = corpus.make_label_dictionary("label", add_unk=True)

    assert len(label_dict) == 3
    assert "<unk>" in label_dict.get_items()
    assert "class_1" in label_dict.get_items()
    assert "class_2" in label_dict.get_items()

    with pytest.warns(DeprecationWarning):  # test to make sure the warning comes, but function works
        corpus.make_tag_dictionary("label")


def test_obtain_statistics():
    sentence_1 = Sentence("The snake hissed to the mountain goat")
    sentence_1_labels = "  O   B-Ani O      O  O   B-Ani    E-Ani".split()
    sentence_2 = Sentence("Saber    tooth tigers are extinct")
    sentence_2_labels = "  B-Ani    I-Ani E-Ani  O   O".split()

    for sentence, labels in [(sentence_1, sentence_1_labels), (sentence_2, sentence_2_labels)]:
        assert len(sentence) == len(labels)
        for token, label in zip(sentence, labels):
            token.add_label("ner", label)
    corpus = Corpus(
        FlairDatapointDataset([sentence_1, sentence_2]),
        FlairDatapointDataset([]),
        FlairDatapointDataset([sentence_2]),
    )
    statistics = corpus.obtain_statistics("ner", pretty_print=False)
    assert statistics == {
        "TRAIN": {
            "dataset": "TRAIN",
            "total_number_of_documents": 2,
            "number_of_documents_per_class": {"O": 6, "B-Ani": 3, "E-Ani": 2, "I-Ani": 1},
            "number_of_tokens_per_tag": {"O": 6, "B-Ani": 3, "E-Ani": 2, "I-Ani": 1},
            "number_of_tokens": {"total": 12, "min": 5, "max": 7, "avg": 6.0},
        },
        "TEST": {
            "dataset": "TEST",
            "total_number_of_documents": 1,
            "number_of_documents_per_class": {"B-Ani": 1, "I-Ani": 1, "E-Ani": 1, "O": 2},
            "number_of_tokens_per_tag": {"B-Ani": 1, "I-Ani": 1, "E-Ani": 1, "O": 2},
            "number_of_tokens": {"total": 5, "min": 5, "max": 5, "avg": 5.0},
        },
        "DEV": {},
    }


def test_tagged_corpus_statistics():
    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label("label", "class_1")

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label("label", "class_2")

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True).add_label("label", "class_1")

    class_to_count_dict = Corpus._count_sentence_labels([train_sentence, dev_sentence, test_sentence])

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert class_to_count_dict["class_1"] == 2
    assert class_to_count_dict["class_2"] == 1

    tokens_in_sentences = Corpus._get_tokens_per_sentence([train_sentence, dev_sentence, test_sentence])

    assert len(tokens_in_sentences) == 3
    assert tokens_in_sentences[0] == 4
    assert tokens_in_sentences[1] == 5
    assert tokens_in_sentences[2] == 4


def test_tagged_corpus_statistics_multi_label():
    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label("label", "class_1")

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label("label", "class_2")

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True)
    test_sentence.add_label("label", "class_1")
    test_sentence.add_label("label", "class_2")

    class_to_count_dict = Corpus._count_sentence_labels([train_sentence, dev_sentence, test_sentence])

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert class_to_count_dict["class_1"] == 2
    assert class_to_count_dict["class_2"] == 2

    tokens_in_sentences = Corpus._get_tokens_per_sentence([train_sentence, dev_sentence, test_sentence])

    assert len(tokens_in_sentences) == 3
    assert tokens_in_sentences[0] == 4
    assert tokens_in_sentences[1] == 5
    assert tokens_in_sentences[2] == 4


def test_tagged_corpus_downsample():
    sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label("label", "class_1")

    corpus: Corpus = Corpus(
        FlairDatapointDataset(
            [
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
                sentence,
            ]
        ),
        sample_missing_splits=False,
    )

    assert len(corpus.train) == 10

    corpus.downsample(percentage=0.3, downsample_dev=False, downsample_test=False)

    assert len(corpus.train) == 3


def test_classification_corpus_multi_labels_without_negative_examples(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(
        tasks_base_path / "multi_class_negative_examples",
        allow_examples_without_labels=False,
    )
    assert len(corpus.train) == 7
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 5


def test_classification_corpus_multi_labels_with_negative_examples(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(
        tasks_base_path / "multi_class_negative_examples",
        allow_examples_without_labels=True,
    )
    assert len(corpus.train) == 8
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 6


def test_misalignment_spans(tasks_base_path):
    example_txt = """George B-NAME
Washington I-NAME
went O
\t O
Washington B-CITY
and O
enjoyed O
some O
coffee B-BEVERAGE
"""
    train_path = tasks_base_path / "tmp" / "train.txt"
    try:
        train_path.parent.mkdir(exist_ok=True, parents=True)
        train_path.write_text(example_txt, encoding="utf-8")
        corpus = ColumnCorpus(
            data_folder=train_path.parent, column_format={0: "text", 1: "ner"}, train_file=train_path.name
        )
        sentence = corpus.train[0]
        span_texts = [span.text for span in sentence.get_spans("ner")]
        assert span_texts == ["George Washington", "Washington", "coffee"]
    finally:
        train_path.unlink()
        train_path.parent.rmdir()
