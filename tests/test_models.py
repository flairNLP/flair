import pytest

import flair
from flair.data import Corpus, Sentence
from flair.datasets import ClassificationCorpus, ColumnCorpus
from flair.embeddings import (
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import Result

turian_embeddings = WordEmbeddings("turian")
flair_embeddings = FlairEmbeddings("news-forward-fast")


@pytest.mark.integration
def test_sequence_tagger_no_crf(results_base_path, tasks_base_path):
    flair.set_seed(123)

    # load dataset
    corpus: Corpus = ColumnCorpus(
        data_folder=tasks_base_path / "trivial" / "trivial_bioes",
        column_format={0: "text", 1: "ner"},
    )
    tag_dictionary = corpus.make_label_dictionary("ner")

    # tagger without CRF
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # train
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=10,
        shuffle=False,
    )

    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("this is New York")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # check if loaded model can predict
    entities = [label.data_point.text for label in sentence.get_labels("ner")]
    assert "New York" in entities

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="ner")
    assert result.classification_report["micro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_sequence_tagger_with_crf(results_base_path, tasks_base_path):
    flair.set_seed(123)

    # load dataset
    corpus: Corpus = ColumnCorpus(
        data_folder=tasks_base_path / "trivial" / "trivial_bioes",
        column_format={0: "text", 1: "ner"},
    )
    tag_dictionary = corpus.make_label_dictionary("ner")

    # tagger without CRF
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=True,
    )

    # train
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=10,
        shuffle=False,
    )

    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("this is New York")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # check if loaded model can predict
    entities = [label.data_point.text for label in sentence.get_labels("ner")]
    assert "New York" in entities

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="ner")
    assert result.classification_report["micro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_sequence_tagger_stacked(results_base_path, tasks_base_path):
    flair.set_seed(123)

    # load dataset
    corpus: Corpus = ColumnCorpus(
        data_folder=tasks_base_path / "trivial" / "trivial_bioes",
        column_format={0: "text", 1: "ner"},
    )
    tag_dictionary = corpus.make_label_dictionary("ner")

    # tagger without CRF
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=StackedEmbeddings([turian_embeddings, flair_embeddings]),
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=True,
    )

    # train
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=10,
        shuffle=False,
    )

    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("this is New York")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # check if loaded model can predict
    entities = [label.data_point.text for label in sentence.get_labels("ner")]
    assert "New York" in entities

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="ner")
    assert result.classification_report["micro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_sequence_tagger_transformer_finetune(results_base_path, tasks_base_path):
    flair.set_seed(123)

    # load dataset
    corpus: Corpus = ColumnCorpus(
        data_folder=tasks_base_path / "trivial" / "trivial_bioes",
        column_format={0: "text", 1: "ner"},
    )
    tag_dictionary = corpus.make_label_dictionary("ner")

    # tagger without CRF
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=TransformerWordEmbeddings("distilbert-base-uncased", fine_tune=True),
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # train
    trainer = ModelTrainer(tagger, corpus)
    trainer.fine_tune(
        results_base_path,
        mini_batch_size=2,
        max_epochs=10,
        shuffle=True,
        learning_rate=0.5e-4,
    )

    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("this is New York")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # check if loaded model can predict
    entities = [label.data_point.text for label in sentence.get_labels("ner")]
    assert "New York" in entities

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="ner")
    assert result.classification_report["micro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_text_classifier(results_base_path, tasks_base_path):
    flair.set_seed(123)

    corpus = ClassificationCorpus(
        tasks_base_path / "trivial" / "trivial_text_classification_single",
        label_type="city",
    )
    label_dict = corpus.make_label_dictionary(label_type="city")

    model: TextClassifier = TextClassifier(
        document_embeddings=DocumentPoolEmbeddings([turian_embeddings]),
        label_dictionary=label_dict,
        label_type="city",
        multi_label=False,
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, mini_batch_size=2, max_epochs=10, shuffle=True)

    # check if model can predict
    sentence = Sentence("this is Berlin")
    sentence_empty = Sentence("       ")

    model.predict(sentence)
    model.predict([sentence, sentence_empty])
    model.predict([sentence_empty])

    # load model
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    # chcek if model predicts correct label
    sentence = Sentence("this is Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict([sentence, sentence_empty])

    values = []
    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float
        values.append(label.value)

    assert "Berlin" in values

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="city")
    assert result.classification_report["macro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_text_classifier_transformer_finetune(results_base_path, tasks_base_path):
    flair.set_seed(123)

    corpus = ClassificationCorpus(
        tasks_base_path / "trivial" / "trivial_text_classification_single",
        label_type="city",
    )
    label_dict = corpus.make_label_dictionary(label_type="city")

    model: TextClassifier = TextClassifier(
        document_embeddings=TransformerDocumentEmbeddings("distilbert-base-uncased"),
        label_dictionary=label_dict,
        label_type="city",
        multi_label=False,
    )

    trainer = ModelTrainer(model, corpus)
    trainer.fine_tune(
        results_base_path,
        mini_batch_size=2,
        max_epochs=10,
        shuffle=True,
        learning_rate=0.5e-5,
        num_workers=2,
    )

    # check if model can predict
    sentence = Sentence("this is Berlin")
    sentence_empty = Sentence("       ")

    model.predict(sentence)
    model.predict([sentence, sentence_empty])
    model.predict([sentence_empty])

    # load model
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    # chcek if model predicts correct label
    sentence = Sentence("this is Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict([sentence, sentence_empty])

    values = []
    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float
        values.append(label.value)

    assert "Berlin" in values

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="city")
    assert result.classification_report["macro avg"]["f1-score"] == 1.0

    del loaded_model


@pytest.mark.integration
def test_text_classifier_multi(results_base_path, tasks_base_path):
    flair.set_seed(123)

    corpus = ClassificationCorpus(
        tasks_base_path / "trivial" / "trivial_text_classification_multi",
        label_type="city",
    )
    label_dict = corpus.make_label_dictionary(label_type="city")

    model: TextClassifier = TextClassifier(
        document_embeddings=DocumentPoolEmbeddings([turian_embeddings], fine_tune_mode="linear"),
        label_dictionary=label_dict,
        label_type="city",
        multi_label=True,
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, mini_batch_size=2, max_epochs=50, shuffle=True)

    # check if model can predict
    sentence = Sentence("this is Berlin")
    sentence_empty = Sentence("       ")

    model.predict(sentence)
    model.predict([sentence, sentence_empty])
    model.predict([sentence_empty])

    # load model
    loaded_model = TextClassifier.load(results_base_path / "final-model.pt")

    # chcek if model predicts correct label
    sentence = Sentence("this is Berlin")
    sentence_double = Sentence("this is Berlin and pizza")

    loaded_model.predict([sentence, sentence_double])

    values = []
    for label in sentence_double.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert type(label.score) is float
        values.append(label.value)

    assert "Berlin" in values
    assert "pizza" in values

    # check if loaded model successfully fit the training data
    result: Result = loaded_model.evaluate(corpus.test, gold_label_type="city")
    print(result.classification_report)
    assert result.classification_report["micro avg"]["f1-score"] == 1.0

    del loaded_model
