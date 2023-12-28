import pytest

from flair.data import Sentence
from flair.datasets import NEL_ENGLISH_AIDA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SpanClassifier
from tests.model_test_utils import BaseModelTest


class TestEntityLinker(BaseModelTest):
    model_cls = SpanClassifier
    train_label_type = "nel"
    training_args = {"max_epochs": 2}

    @pytest.fixture()
    def embeddings(self):
        return TransformerWordEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return NEL_ENGLISH_AIDA().downsample(0.01)

    @pytest.fixture()
    def train_test_sentence(self):
        sentence = Sentence("I love NYC and hate OYC")

        sentence[2:3].add_label("nel", "New York City")
        sentence[5:6].add_label("nel", "Old York City")
        return sentence

    @pytest.fixture()
    def labeled_sentence(self):
        sentence = Sentence("I love NYC and hate OYC")

        sentence[2:3].add_label("nel", "New York City")
        sentence[5:6].add_label("nel", "Old York City")
        return sentence
