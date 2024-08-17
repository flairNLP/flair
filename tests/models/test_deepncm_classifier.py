import pytest
import torch

from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import DeepNCMClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins import DeepNCMPlugin
from tests.model_test_utils import BaseModelTest


class TestDeepNCMClassifier(BaseModelTest):
    model_cls = DeepNCMClassifier
    train_label_type = "class"
    multiclass_prediction_labels = ["POSITIVE", "NEGATIVE"]
    training_args = {
        "max_epochs": 2,
        "mini_batch_size": 4,
        "learning_rate": 1e-5,
    }

    @pytest.fixture()
    def embeddings(self):
        return TransformerDocumentEmbeddings("distilbert-base-uncased", fine_tune=True)

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return ClassificationCorpus(tasks_base_path / "imdb", label_type=self.train_label_type)

    @pytest.fixture()
    def multiclass_train_test_sentence(self):
        return Sentence("This movie was great!")

    def build_model(self, embeddings, label_dict, **kwargs):
        model_args = {
            "embeddings": embeddings,
            "label_dictionary": label_dict,
            "label_type": self.train_label_type,
            "use_encoder": False,
            "encoding_dim": 64,
            "alpha": 0.95,
        }
        model_args.update(kwargs)
        return self.model_cls(**model_args)

    @pytest.mark.integration()
    def test_train_load_use_classifier(
        self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence
    ):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)

        model = self.build_model(embeddings, label_dict, mean_update_method="condensation")

        trainer = ModelTrainer(model, corpus)
        trainer.fine_tune(
            results_base_path, optimizer=torch.optim.AdamW, plugins=[DeepNCMPlugin()], **self.training_args
        )

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

    def test_get_prototype(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)

        prototype = model.get_prototype(next(iter(label_dict.get_items())))
        assert isinstance(prototype, torch.Tensor)
        assert prototype.shape == (model.encoding_dim,)

        with pytest.raises(ValueError):
            model.get_prototype("NON_EXISTENT_CLASS")

    def test_get_closest_prototypes(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)
        input_vector = torch.randn(model.encoding_dim)
        closest_prototypes = model.get_closest_prototypes(input_vector, top_k=2)

        assert len(closest_prototypes) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in closest_prototypes)

        with pytest.raises(ValueError):
            model.get_closest_prototypes(torch.randn(model.encoding_dim + 1))

    def test_forward_loss(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)

        sentences = [Sentence("This movie was great!"), Sentence("I didn't enjoy this film at all.")]
        for sentence, label in zip(sentences, list(label_dict.get_items())[:2]):
            sentence.add_label(self.train_label_type, label)

        loss, count = model.forward_loss(sentences)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert count == len(sentences)

    @pytest.mark.parametrize("mean_update_method", ["online", "condensation", "decay"])
    def test_mean_update_methods(self, corpus, embeddings, mean_update_method):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict, mean_update_method=mean_update_method)

        initial_prototypes = model.class_prototypes.clone()

        sentences = [Sentence("This movie was great!"), Sentence("I didn't enjoy this film at all.")]
        for sentence, label in zip(sentences, list(label_dict.get_items())[:2]):
            sentence.add_label(self.train_label_type, label)

        model.forward_loss(sentences)
        model.update_prototypes()

        assert not torch.all(torch.eq(initial_prototypes, model.class_prototypes))

    @pytest.mark.parametrize("mean_update_method", ["online", "condensation", "decay"])
    def test_deepncm_plugin(self, corpus, embeddings, mean_update_method):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict, mean_update_method=mean_update_method)

        trainer = ModelTrainer(model, corpus)
        plugin = DeepNCMPlugin()
        plugin.attach_to(trainer)

        initial_class_counts = model.class_counts.clone()
        initial_prototypes = model.class_prototypes.clone()

        # Simulate training epoch
        plugin.after_training_epoch()

        if mean_update_method == "condensation":
            assert torch.all(model.class_counts == 1), "Class counts should be 1 for condensation method after epoch"
        elif mean_update_method == "online":
            assert torch.all(
                torch.eq(model.class_counts, initial_class_counts)
            ), "Class counts should not change for online method after epoch"

        # Simulate training batch
        sentences = [Sentence("This movie was great!"), Sentence("I didn't enjoy this film at all.")]
        for sentence, label in zip(sentences, list(label_dict.get_items())[:2]):
            sentence.add_label(self.train_label_type, label)
        model.forward_loss(sentences)
        plugin.after_training_batch()

        assert not torch.all(
            torch.eq(initial_prototypes, model.class_prototypes)
        ), "Prototypes should be updated after a batch"

        if mean_update_method == "condensation":
            assert torch.all(
                model.class_counts >= 1
            ), "Class counts should be >= 1 for condensation method after a batch"
        elif mean_update_method == "online":
            assert torch.all(
                model.class_counts > initial_class_counts
            ), "Class counts should increase for online method after a batch"
