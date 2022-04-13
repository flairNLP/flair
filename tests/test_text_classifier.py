import pytest

import flair.datasets
from flair.data import Sentence
from flair.embeddings import DocumentRNNEmbeddings, FlairEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.samplers import ImbalancedClassificationDatasetSampler
from flair.trainers import ModelTrainer

turian_embeddings = WordEmbeddings("turian")
flair_embeddings = FlairEmbeddings("news-forward-fast")
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings([turian_embeddings], 128, 1, False, 64, False, False)


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
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    model: TextClassifier = TextClassifier(
        document_embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type="topic",
        multi_label=False,
    )

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

    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_with_sampler(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    model: TextClassifier = TextClassifier(
        document_embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type="topic",
        multi_label=False,
    )

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

    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_with_prob(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    model: TextClassifier = TextClassifier(
        document_embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type="topic",
        multi_label=False,
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)

    sentence = Sentence("Berlin is a really nice city.")

    model.predict(sentence, return_probabilities_for_all_classes=True)

    assert len(sentence.labels) > 1

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence, return_probabilities_for_all_classes=True)
    loaded_model.predict([sentence, sentence_empty], return_probabilities_for_all_classes=True)
    loaded_model.predict([sentence_empty], return_probabilities_for_all_classes=True)

    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_multi_label(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "multi_class", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    model: TextClassifier = TextClassifier(
        document_embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type="topic",
        multi_label=True,
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(
        results_base_path,
        mini_batch_size=1,
        max_epochs=20,
        shuffle=False,
        checkpoint=False,
        train_with_test=True,
        train_with_dev=True,
    )

    sentence = Sentence("apple tv")

    model.predict(sentence)

    assert "apple" in [label.value for label in sentence.labels]
    assert "tv" in [label.value for label in sentence.labels]

    for label in sentence.labels:
        print(label)
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    del trainer, model, corpus
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    sentence = Sentence("apple tv")

    loaded_model.predict(sentence)

    assert "apple" in [label.value for label in sentence.labels]
    assert "tv" in [label.value for label in sentence.labels]

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_load_use_classifier_flair(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    flair_document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [flair_embeddings], 128, 1, False, 64, False, False
    )

    model: TextClassifier = TextClassifier(
        document_embeddings=flair_document_embeddings,
        label_dictionary=label_dict,
        label_type="topic",
        multi_label=False,
    )

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

    del loaded_model


@pytest.mark.integration
def test_train_resume_classifier(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")
    label_dict = corpus.make_label_dictionary(label_type="topic")

    model = TextClassifier(
        document_embeddings=document_embeddings,
        label_dictionary=label_dict,
        multi_label=False,
        label_type="topic",
    )

    # train model for 2 epochs
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False, checkpoint=True)

    del model

    # load the checkpoint model and train until epoch 4
    checkpoint_model = TextClassifier.load(results_base_path / "checkpoint.pt")
    with pytest.warns(UserWarning):
        trainer.resume(model=checkpoint_model, max_epochs=4)

    del trainer
