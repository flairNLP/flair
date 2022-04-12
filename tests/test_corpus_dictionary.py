import os

import flair
from flair.data import Corpus, Dictionary, Label, Sentence
from flair.datasets import FlairDatapointDataset


def test_dictionary_get_items_with_unk():
    dictionary: Dictionary = Dictionary()

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert 4 == len(items)
    assert "<unk>" == items[0]
    assert "class_1" == items[1]
    assert "class_2" == items[2]
    assert "class_3" == items[3]


def test_dictionary_get_items_without_unk():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert 3 == len(items)
    assert "class_1" == items[0]
    assert "class_2" == items[1]
    assert "class_3" == items[2]


def test_dictionary_get_idx_for_item():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    idx = dictionary.get_idx_for_item("class_2")

    assert 1 == idx


def test_dictionary_get_item_for_index():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    item = dictionary.get_item_for_index(0)

    assert "class_1" == item


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

    assert 3 == len(all_sentences)


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence("used in training. training is cool.")

    corpus: Corpus = Corpus(FlairDatapointDataset([train_sentence]), sample_missing_splits=False)

    vocab = corpus.make_vocab_dictionary(max_tokens=2, min_freq=-1)

    assert 3 == len(vocab)
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=-1)

    assert 7 == len(vocab)

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=2)

    assert 3 == len(vocab)
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()


def test_label_set_confidence():
    label = Label(data_point=None, value="class_1", score=3.2)

    assert 3.2 == label.score
    assert "class_1" == label.value

    label.score = 0.2

    assert 0.2 == label.score


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence("sentence 1").add_label("label", "class_1")

    sentence_2 = Sentence("sentence 2").add_label("label", "class_2")

    sentence_3 = Sentence("sentence 3").add_label("label", "class_1")

    corpus: Corpus = Corpus(
        FlairDatapointDataset([sentence_1, sentence_2, sentence_3]),
        FlairDatapointDataset([]),
        FlairDatapointDataset([]),
    )

    label_dict = corpus.make_label_dictionary("label")

    assert 3 == len(label_dict)
    assert "<unk>" in label_dict.get_items()
    assert "class_1" in label_dict.get_items()
    assert "class_2" in label_dict.get_items()


def test_tagged_corpus_statistics():
    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label("label", "class_1")

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label("label", "class_2")

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True).add_label("label", "class_1")

    class_to_count_dict = Corpus._count_sentence_labels([train_sentence, dev_sentence, test_sentence])

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert 2 == class_to_count_dict["class_1"]
    assert 1 == class_to_count_dict["class_2"]

    tokens_in_sentences = Corpus._get_tokens_per_sentence([train_sentence, dev_sentence, test_sentence])

    assert 3 == len(tokens_in_sentences)
    assert 4 == tokens_in_sentences[0]
    assert 5 == tokens_in_sentences[1]
    assert 4 == tokens_in_sentences[2]


def test_tagged_corpus_statistics_multi_label():
    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label("label", "class_1")

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label("label", "class_2")

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True)
    test_sentence.add_label("label", "class_1")
    test_sentence.add_label("label", "class_2")

    class_to_count_dict = Corpus._count_sentence_labels([train_sentence, dev_sentence, test_sentence])

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert 2 == class_to_count_dict["class_1"]
    assert 2 == class_to_count_dict["class_2"]

    tokens_in_sentences = Corpus._get_tokens_per_sentence([train_sentence, dev_sentence, test_sentence])

    assert 3 == len(tokens_in_sentences)
    assert 4 == tokens_in_sentences[0]
    assert 5 == tokens_in_sentences[1]
    assert 4 == tokens_in_sentences[2]


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

    assert 10 == len(corpus.train)

    corpus.downsample(percentage=0.3, downsample_dev=False, downsample_test=False)

    assert 3 == len(corpus.train)


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
