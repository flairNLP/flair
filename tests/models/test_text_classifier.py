import pytest

import flair.datasets
from flair.data import Sentence
from flair.embeddings import DocumentRNNEmbeddings, FlairEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.samplers import ImbalancedClassificationDatasetSampler
from flair.trainers import ModelTrainer
from tests.model_test_utils import BaseModelTest


class TestTextClassifier(BaseModelTest):
    model_cls = TextClassifier
    pretrained_model = "sentiment"
    train_label_type = "topic"
    multiclass_prediction_labels = ["apple", "tv"]
    training_args = {
        "max_epochs": 4,
    }

    @pytest.fixture()
    def embeddings(self):
        turian_embeddings = WordEmbeddings("turian")
        document_embeddings = DocumentRNNEmbeddings([turian_embeddings], 128, 1, False, 64, False, False)
        return document_embeddings

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="topic")

    @pytest.fixture()
    def multiclass_train_test_sentence(self):
        return Sentence("apple tv")

    @pytest.fixture()
    def multi_class_corpus(self, tasks_base_path):
        return flair.datasets.ClassificationCorpus(tasks_base_path / "multi_class", label_type="topic")

    @pytest.mark.integration()
    def test_train_load_use_classifier_with_sampler(
        self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence
    ):
        flair.set_seed(123)
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.model_cls(embeddings=embeddings, label_dictionary=label_dict, label_type=self.train_label_type)

        trainer = ModelTrainer(model, corpus)
        trainer.train(results_base_path, max_epochs=2, shuffle=False, sampler=ImbalancedClassificationDatasetSampler)

        model.predict(train_test_sentence)

        for label in train_test_sentence.get_labels(self.train_label_type):
            assert label.value is not None
            assert 0.0 <= label.score <= 1.0
            assert isinstance(label.score, float)

        del trainer, model, corpus

        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])

    @pytest.mark.integration()
    def test_predict_with_prob(self, example_sentence, loaded_pretrained_model):
        loaded_pretrained_model.predict(example_sentence, return_probabilities_for_all_classes=True)
        assert len(example_sentence.get_labels(loaded_pretrained_model.label_type)) == len(
            loaded_pretrained_model.label_dictionary
        )
        assert (
            sum([label.score for label in example_sentence.get_labels(loaded_pretrained_model.label_type)]) > 1 - 1e-5
        )

    @pytest.mark.integration()
    def test_train_load_use_classifier_flair(self, results_base_path, corpus, example_sentence, train_test_sentence):
        flair.set_seed(123)
        embeddings = DocumentRNNEmbeddings([FlairEmbeddings("news-forward-fast")], 128, 1, False, 64, False, False)
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.model_cls(embeddings=embeddings, label_dictionary=label_dict, label_type=self.train_label_type)

        trainer = ModelTrainer(model, corpus)
        trainer.train(results_base_path, max_epochs=2, shuffle=False)

        model.predict(train_test_sentence)

        for label in train_test_sentence.get_labels(self.train_label_type):
            assert label.value is not None
            assert 0.0 <= label.score <= 1.0
            assert isinstance(label.score, float)

        del trainer, model, corpus

        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")
        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
