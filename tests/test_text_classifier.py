import flair.datasets
import shutil

import pytest

import flair.datasets
from flair.data import Sentence
from flair.embeddings import (
    WordEmbeddings,
    FlairEmbeddings,
    DocumentRNNEmbeddings,
)
from flair.models import TextClassifier
from flair.samplers import ImbalancedClassificationDatasetSampler
from flair.trainers import ModelTrainer

turian_embeddings = WordEmbeddings("turian")
flair_embeddings = FlairEmbeddings("news-forward-fast")
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
    [turian_embeddings], 128, 1, False, 64, False, False
)


@pytest.mark.integration
def test_load_use_classifier():
    loaded_model: TextClassifier = TextClassifier.load("sentiment")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    del loaded_model

    sentence.clear_embeddings()
    sentence_empty.clear_embeddings()


@pytest.mark.integration
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")
    label_dict = corpus.make_label_dictionary()

    model: TextClassifier = TextClassifier(document_embeddings, label_dict, multi_label=False)

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)

    sentence = Sentence("Berlin is a really nice city.")

    model.predict(sentence)

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_with_sampler(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")
    label_dict = corpus.make_label_dictionary()

    model: TextClassifier = TextClassifier(document_embeddings, label_dict, multi_label=False)

    trainer = ModelTrainer(model, corpus)
    trainer.train(
        results_base_path,
        max_epochs=2,
        shuffle=False,
        sampler=ImbalancedClassificationDatasetSampler,
    )

    sentence = Sentence("Berlin is a really nice city.")
    model.predict(sentence)

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_with_prob(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")
    label_dict = corpus.make_label_dictionary()

    model: TextClassifier = TextClassifier(document_embeddings, label_dict, multi_label=False)

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)

    sentence = Sentence("Berlin is a really nice city.")

    model.predict(sentence, multi_class_prob=True)

    assert len(sentence.labels) > 1

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence, multi_class_prob=True)
    loaded_model.predict([sentence, sentence_empty], multi_class_prob=True)
    loaded_model.predict([sentence_empty], multi_class_prob=True)

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_multi_label(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "multi_class")
    label_dict = corpus.make_label_dictionary()

    model: TextClassifier = TextClassifier(
        document_embeddings, label_dict, multi_label=True
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(
        results_base_path,
        mini_batch_size=1,
        max_epochs=100,
        shuffle=False,
        checkpoint=False,
    )

    sentence = Sentence("apple tv")

    model.predict(sentence)

    for label in sentence.labels:
        print(label)
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    sentence = Sentence("apple tv")

    model.predict(sentence)

    assert "apple" in sentence.get_label_names()
    assert "tv" in sentence.get_label_names()

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_flair(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")
    label_dict = corpus.make_label_dictionary()

    flair_document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
       [flair_embeddings], 128, 1, False, 64, False, False
    )

    model: TextClassifier = TextClassifier(flair_document_embeddings, label_dict, multi_label=False)

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)

    sentence = Sentence("Berlin is a really nice city.")

    model.predict(sentence)

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus, flair_document_embeddings
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model


@pytest.mark.integration
def test_train_resume_classifier(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")
    label_dict = corpus.make_label_dictionary()

    model = TextClassifier(document_embeddings, label_dict, multi_label=False)

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False, checkpoint=True)

    del trainer, model
    trainer = ModelTrainer.load_checkpoint(results_base_path / "checkpoint.pt", corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False, checkpoint=True)

    # clean up results directory
    shutil.rmtree(results_base_path)
    del trainer


def test_labels_to_indices(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "ag_news")
    label_dict = corpus.make_label_dictionary()
    model = TextClassifier(document_embeddings, label_dict, multi_label=False)

    result = model._labels_to_indices(corpus.train)

    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i].item()

        assert expected == actual


def test_labels_to_one_hot(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "ag_news")
    label_dict = corpus.make_label_dictionary()
    model = TextClassifier(document_embeddings, label_dict, multi_label=False)

    result = model._labels_to_one_hot(corpus.train)

    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i]

        for idx in range(len(label_dict)):
            if idx == expected:
                assert actual[idx] == 1
            else:
                assert actual[idx] == 0