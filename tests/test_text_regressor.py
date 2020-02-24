import pytest
from typing import Tuple
import flair.datasets
from flair.data import Dictionary, Corpus
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models.text_regression_model import TextRegressor

# from flair.trainers.trainer_regression import RegressorTrainer
from flair.trainers import ModelTrainer


def init(tasks_base_path) -> Tuple[Corpus, TextRegressor, ModelTrainer]:
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / 'regression')

    glove_embedding: WordEmbeddings = WordEmbeddings("glove")
    document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False
    )

    model = TextRegressor(document_embeddings)

    trainer = ModelTrainer(model, corpus)

    return corpus, model, trainer


def test_labels_to_indices(tasks_base_path):
    corpus, model, trainer = init(tasks_base_path)

    result = model._labels_to_indices(corpus.train)

    for i in range(len(corpus.train)):
        expected = round(float(corpus.train[i].labels[0].value), 3)
        actual = round(float(result[i].item()), 3)

        assert expected == actual


def test_trainer_evaluation(tasks_base_path):
    corpus, model, trainer = init(tasks_base_path)

    expected = model.evaluate(corpus.dev)

    assert expected is not None


# def test_trainer_results(tasks_base_path):
#    corpus, model, trainer = init(tasks_base_path)

#    results = trainer.train("regression_train/", max_epochs=1)

#    assert results["test_score"] > 0
#    assert len(results["dev_loss_history"]) == 1
#    assert len(results["dev_score_history"]) == 1
#    assert len(results["train_loss_history"]) == 1
