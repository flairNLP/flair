import pytest

import flair
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TokenClassifier
from tests.model_test_utils import BaseModelTest


class TestWordTagger(BaseModelTest):
    model_cls = TokenClassifier
    train_label_type = "pos"
    training_args = {
        "max_epochs": 2,
        "learning_rate": 0.1,
        "mini_batch_size": 2,
    }

    def has_embedding(self, sentence):
        for token in sentence:
            if token.get_embedding().cpu().numpy().size == 0:
                return False
        return None

    def build_model(self, embeddings, label_dict, **kwargs):
        model_args = dict(self.model_args)
        for k in kwargs:
            if k in model_args:
                del model_args[k]
        return self.model_cls(
            embeddings=embeddings,
            label_dictionary=label_dict,
            label_type=self.train_label_type,
            **model_args,
            **kwargs,
        )

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return flair.datasets.UD_ENGLISH(tasks_base_path)

    @pytest.fixture()
    def embeddings(self):
        return TransformerWordEmbeddings("distilbert-base-uncased")
