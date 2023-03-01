import pytest

import flair
from flair.embeddings import TransformerWordEmbeddings
from flair.models import WordTagger
from tests.model_test_utils import BaseModelTest


class TestWordTagger(BaseModelTest):
    model_cls = WordTagger
    train_label_type = "pos"
    training_args = dict(
        max_epochs=2,
        learning_rate=0.1,
        mini_batch_size=2,
    )

    def has_embedding(self, sentence):
        for token in sentence:
            if token.get_embedding().cpu().numpy().size == 0:
                return False
        return

    def build_model(self, embeddings, label_dict, **kwargs):
        model_args = dict(self.model_args)
        for k in kwargs.keys():
            if k in model_args:
                del model_args[k]
        return self.model_cls(
            embeddings=embeddings,
            tag_dictionary=label_dict,
            tag_type=self.train_label_type,
            **model_args,
            **kwargs,
        )

    @pytest.fixture
    def corpus(self, tasks_base_path):
        yield flair.datasets.UD_ENGLISH(tasks_base_path)

    @pytest.fixture
    def embeddings(self):
        yield TransformerWordEmbeddings("distilbert-base-uncased")
