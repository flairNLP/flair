import pytest

from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import RelationExtractor
from tests.model_test_utils import BaseModelTest


class TestRelationExtractor(BaseModelTest):
    model_cls = RelationExtractor
    train_label_type = "relation"
    pretrained_model = "relations"
    model_args = {
        "entity_label_type": "ner",
        "train_on_gold_pairs_only": True,
        "entity_pair_filters": {  # Define valid entity pair combinations, used as relation candidates
            ("ORG", "PER"),  # founded_by
            ("LOC", "PER"),  # place_of_birth
        },
    }
    training_args = {
        "max_epochs": 4,
        "mini_batch_size": 2,
        "learning_rate": 0.1,
    }

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return ColumnCorpus(
            data_folder=tasks_base_path / "conllu",
            train_file="train.conllup",
            dev_file="train.conllup",
            test_file="train.conllup",
            column_format={1: "text", 2: "pos", 3: "ner"},
        )

    @pytest.fixture()
    def example_sentence(self):
        sentence = Sentence(["Microsoft", "was", "found", "by", "Bill", "Gates"])
        sentence[:1].add_label(typename="ner", value="ORG", score=1.0)
        sentence[4:].add_label(typename="ner", value="PER", score=1.0)
        return sentence

    @pytest.fixture()
    def train_test_sentence(self):
        sentence = Sentence(["Apple", "was", "founded", "by", "Steve", "Jobs", "."])
        sentence[0:1].add_label("ner", "ORG")
        sentence[4:6].add_label("ner", "PER")
        return sentence

    @pytest.fixture()
    def embeddings(self):
        return TransformerWordEmbeddings(model="distilbert-base-uncased", fine_tune=True)

    def assert_training_example(self, predicted_training_example):
        relations = predicted_training_example.get_relations("relation")
        assert len(relations) == 1
        assert relations[0].tag == "founded_by"

    def has_embedding(self, sentence):
        return all(token.get_embedding().cpu().numpy().size != 0 for token in sentence)
