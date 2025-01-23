import pytest

import flair
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models.one_class_classification_model import OneClassClassifier
from flair.trainers import ModelTrainer
from tests.model_test_utils import BaseModelTest


class TestOneClassClassifier(BaseModelTest):
    model_cls = OneClassClassifier
    train_label_type = "topic"
    training_args = {
        "max_epochs": 2,
    }

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        label_type = "topic"
        corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type=label_type)
        corpus._train = [x for x in corpus.train if x.get_labels(label_type)[0].value == "POSITIVE"]
        return corpus

    @pytest.fixture()
    def embeddings(self):
        return TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    @pytest.mark.integration()
    def test_train_load_use_one_class_classifier(self, results_base_path, corpus, example_sentence, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.model_cls(embeddings=embeddings, label_dictionary=label_dict, label_type=self.train_label_type)
        trainer = ModelTrainer(model, corpus)

        trainer.train(results_base_path, shuffle=False, **self.training_args)

        del trainer, model, label_dict, corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])

        assert example_sentence.get_labels(self.train_label_type)[0].value in {"POSITIVE", "<unk>"}
