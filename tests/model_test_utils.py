from typing import Any, Optional

import pytest

import flair
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerEmbeddings
from flair.models import FewshotClassifier
from flair.nn import Model
from flair.trainers import ModelTrainer


class BaseModelTest:
    model_cls: type[Model]
    pretrained_model: Optional[str] = None
    empty_sentence = Sentence("       ")
    train_label_type: str
    multiclass_prediction_labels: list[str]
    model_args: dict[str, Any] = {}
    training_args: dict[str, Any] = {}
    finetune_instead_of_train: bool = False

    @pytest.fixture()
    def embeddings(self):
        pytest.skip("This test requires the `embeddings` fixture to be defined")

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        pytest.skip("This test requires the `corpus` fixture to be defined")

    @pytest.fixture()
    def multi_class_corpus(self, tasks_base_path):
        pytest.skip("This test requires the `multi_class_corpus` fixture to be defined")

    @pytest.fixture()
    def multi_corpus(self, tasks_base_path):
        pytest.skip("This test requires the `multi_corpus` fixture to be defined")

    @pytest.fixture()
    def example_sentence(self):
        return Sentence("I love Berlin")

    @pytest.fixture()
    def train_test_sentence(self):
        return Sentence("Berlin is a really nice city.")

    @pytest.fixture()
    def labeled_sentence(self):
        pytest.skip("This test requires the `labeled_sentence` fixture to be defined")

    @pytest.fixture()
    def multiclass_train_test_sentence(self):
        pytest.skip("This test requires the `multiclass_train_test_sentence` fixture to be defined")

    def transform_corpus(self, model, corpus):
        return corpus

    def assert_training_example(self, predicted_training_example):
        pass

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

    def has_embedding(self, sentence):
        return sentence.get_embedding().cpu().numpy().size > 0

    @pytest.fixture()
    def loaded_pretrained_model(self):
        if self.pretrained_model is None:
            pytest.skip("For this test `pretrained_model` needs to be set.")
        return self.model_cls.load(self.pretrained_model)

    @pytest.mark.integration()
    def test_load_use_model(self, example_sentence, loaded_pretrained_model):
        loaded_pretrained_model.predict(example_sentence)
        loaded_pretrained_model.predict([example_sentence, self.empty_sentence])
        loaded_pretrained_model.predict([self.empty_sentence])
        del loaded_pretrained_model

        example_sentence.clear_embeddings()
        self.empty_sentence.clear_embeddings()

    @pytest.mark.integration()
    def test_train_load_use_model(self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence):
        flair.set_seed(123)
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.build_model(embeddings, label_dict)
        corpus = self.transform_corpus(model, corpus)

        trainer = ModelTrainer(model, corpus)

        if self.finetune_instead_of_train:
            trainer.fine_tune(results_base_path, shuffle=False, **self.training_args)
        else:
            trainer.train(results_base_path, shuffle=False, **self.training_args)

        model.predict(train_test_sentence)
        self.assert_training_example(train_test_sentence)

        for label in train_test_sentence.get_labels(self.train_label_type):
            assert label.value is not None
            assert 0.0 <= label.score <= 1.0
            assert isinstance(label.score, float)

        del trainer, model, corpus

        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

    @pytest.mark.integration()
    def test_train_load_use_model_multi_corpus(
        self, results_base_path, multi_corpus, embeddings, example_sentence, train_test_sentence
    ):
        flair.set_seed(123)
        label_dict = multi_corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.build_model(embeddings, label_dict)
        corpus = self.transform_corpus(model, multi_corpus)

        trainer = ModelTrainer(model, corpus)

        if self.finetune_instead_of_train:
            trainer.fine_tune(results_base_path, shuffle=False, **self.training_args)
        else:
            trainer.train(results_base_path, shuffle=False, **self.training_args)

        model.predict(train_test_sentence)
        self.assert_training_example(train_test_sentence)

        for label in train_test_sentence.get_labels(self.train_label_type):
            assert label.value is not None
            assert 0.0 <= label.score <= 1.0
            assert isinstance(label.score, float)

        del trainer, model, corpus

        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

    def test_forward_loss(self, labeled_sentence, embeddings):
        label_dict = Dictionary()
        for label in labeled_sentence.get_labels(self.train_label_type):
            label_dict.add_item(label.value)
        model = self.build_model(embeddings, label_dict)

        loss, count = model.forward_loss([labeled_sentence])
        assert loss.size() == ()
        assert count == len(labeled_sentence.get_labels(self.train_label_type))

    def test_load_use_model_keep_embedding(self, example_sentence, loaded_pretrained_model):
        assert not self.has_embedding(example_sentence)

        loaded_pretrained_model.predict(example_sentence, embedding_storage_mode="cpu")
        assert self.has_embedding(example_sentence)
        del loaded_pretrained_model

    def test_train_load_use_model_multi_label(
        self, results_base_path, multi_class_corpus, embeddings, example_sentence, multiclass_train_test_sentence
    ):
        flair.set_seed(123)
        label_dict = multi_class_corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.build_model(embeddings, label_dict, multi_label=True)
        corpus = self.transform_corpus(model, multi_class_corpus)

        trainer = ModelTrainer(model, corpus)
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
            assert isinstance(label.score, float)

        del trainer, model, multi_class_corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])

    def test_context_is_set_correctly(self):
        sentences = [
            Sentence("this is a very very very long sentence"),
            Sentence("this is a shorter sentence"),
            Sentence(""),
            Sentence("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            Sentence("b"),
        ]
        embedding = TransformerEmbeddings("distilbert-base-cased", use_context=True)
        label_dictionary = Dictionary()
        model = self.build_model(embedding, label_dictionary)

        if isinstance(model, FewshotClassifier):
            model.add_and_switch_to_new_task("test", ["a", "b"], "label")

        model.predict(sentences)

        for first, second in zip(sentences[:-1], sentences[1:]):
            assert first.next_sentence() == second
            assert first == second.previous_sentence()

        assert sentences[0].previous_sentence() is None
        assert sentences[-1].next_sentence() is None
