import pytest

import flair
from flair.embeddings import DocumentRNNEmbeddings, WordEmbeddings
from flair.models.text_regression_model import TextRegressor
from tests.model_test_utils import BaseModelTest


class TestTextRegressor(BaseModelTest):
    model_cls = TextRegressor
    train_label_type = "regression"
    training_args = {
        "max_epochs": 3,
        "mini_batch_size": 2,
        "learning_rate": 0.1,
        "main_evaluation_metric": ("correlation", "pearson"),
    }

    def build_model(self, embeddings, label_dict, **kwargs):
        # no need for label_dict
        return self.model_cls(embeddings, self.train_label_type)

    @pytest.fixture()
    def embeddings(self):
        glove_embedding = WordEmbeddings("turian")
        return DocumentRNNEmbeddings([glove_embedding], 128, 1, False, 64, False, False)

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return flair.datasets.ClassificationCorpus(tasks_base_path / "regression", label_type=self.train_label_type)
