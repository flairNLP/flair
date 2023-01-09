import pytest

from flair.data import Sentence
from flair.datasets import NEL_ENGLISH_AIDA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import EntityLinker
from tests.model_test_utils import BaseModelTest


class TestEntityLinker(BaseModelTest):
    model_cls = EntityLinker
    train_label_type = "nel"
    training_args = dict(max_epochs=5)
    model_args = dict(input_span_label_type="ner")
    finetune_instead_of_train = True

    @pytest.fixture
    def embeddings(self):
        yield TransformerWordEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    def assert_training_example(self, predicted_training_example):
        assert predicted_training_example[1:2].get_label("nel").value == "Amsterdam"

    @pytest.fixture
    def corpus(self, tasks_base_path):
        import random

        random.seed(42)
        yield NEL_ENGLISH_AIDA().downsample(0.05)

    @pytest.fixture
    def train_test_sentence(self):
        sentence = Sentence("-- Amsterdam newsroom +31 20 504 5000")
        sentence[1:2].add_label("ner", "M")
        return sentence

    @pytest.fixture
    def labeled_sentence(self):
        sentence = Sentence("I love NYC and hate OYC")

        sentence[2:3].add_label("nel", "New York City")
        sentence[5:6].add_label("nel", "Old York City")
        return sentence
