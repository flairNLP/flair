from typing import List, Type

import pytest

import flair
from flair.data import Sentence
from flair.nn import Model
from flair.trainers import ModelTrainer


class BaseModelTest:
    model_cls: Type[Model]
    load_model: str
    empty_sentence = Sentence("       ")
    train_label_type: str
    multiclass_prediction_labels: List[str]

    @pytest.fixture
    def embeddings(self):
        pytest.skip("This test requires the `embeddings` fixture to be defined")

    @pytest.fixture
    def corpus(self, tasks_base_path):
        pytest.skip("This test requires the `corpus` fixture to be defined")

    @pytest.fixture
    def multi_class_corpus(self, tasks_base_path):
        pytest.skip("This test requires the `multi_class_corpus` fixture to be defined")

    @pytest.fixture
    def example_sentence(self):
        yield Sentence("I love Berlin")

    @pytest.fixture
    def train_test_sentence(self):
        yield Sentence("Berlin is a really nice city.")

    @pytest.fixture
    def multiclass_train_test_sentence(self):
        pytest.skip("This test requires the `multiclass_train_test_sentence` fixture to be defined")

    @pytest.mark.integration
    def test_load_use_model(self, example_sentence):
        loaded_model = self.model_cls.load(self.load_model)

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

        example_sentence.clear_embeddings()
        self.empty_sentence.clear_embeddings()

    @pytest.mark.integration
    def test_train_load_use_model(self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence):
        flair.set_seed(123)
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

    @pytest.mark.integration
    def test_train_resume_classifier(
        self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence
    ):
        flair.set_seed(123)
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.model_cls(embeddings=embeddings, label_dictionary=label_dict, label_type=self.train_label_type)

        trainer = ModelTrainer(model, corpus)
        trainer.train(results_base_path, max_epochs=2, shuffle=False, checkpoint=True)

        del model
        checkpoint_model = self.model_cls.load(results_base_path / "checkpoint.pt")
        with pytest.warns(UserWarning):
            trainer.resume(model=checkpoint_model, max_epochs=4)
        checkpoint_model.predict(train_test_sentence)

        del trainer, checkpoint_model, corpus

    def test_train_load_use_model_multi_label(
        self, results_base_path, multi_class_corpus, embeddings, example_sentence, multiclass_train_test_sentence
    ):
        flair.set_seed(123)
        label_dict = multi_class_corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.model_cls(
            embeddings=embeddings, label_dictionary=label_dict, label_type=self.train_label_type, multi_label=True
        )

        trainer = ModelTrainer(model, multi_class_corpus)
        trainer.train(
            results_base_path,
            mini_batch_size=1,
            max_epochs=5,
            shuffle=False,
            train_with_test=True,
            train_with_dev=True,
        )

        model.predict(multiclass_train_test_sentence)

        sentence = Sentence("apple tv")

        model.predict(sentence)
        for label in self.multiclass_prediction_labels:
            assert label in [label.value for label in sentence.get_labels(self.train_label_type)], label

        for label in sentence.labels:
            print(label)
            assert label.value is not None
            assert 0.0 <= label.score <= 1.0
            assert type(label.score) is float

        del trainer, model, multi_class_corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
