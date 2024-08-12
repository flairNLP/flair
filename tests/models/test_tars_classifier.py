import pytest

from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TARSClassifier
from tests.model_test_utils import BaseModelTest


class TestTarsClassifier(BaseModelTest):
    model_cls = TARSClassifier
    train_label_type = "class"
    model_args = {"task_name": "2_CLASS"}
    training_args = {"mini_batch_size": 1, "max_epochs": 2}
    # pretrained_model = "tars-base"  # disabled due to too much space requirements.

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return ClassificationCorpus(tasks_base_path / "imdb_underscore")

    @pytest.fixture()
    def embeddings(self):
        return TransformerDocumentEmbeddings("distilbert-base-uncased")

    @pytest.fixture()
    def example_sentence(self):
        return Sentence("This is great!")

    def build_model(self, embeddings, label_dict, **kwargs):
        model_args = dict(self.model_args)
        for k in kwargs:
            if k in model_args:
                del model_args[k]
        return self.model_cls(
            embeddings=embeddings,
            label_type=self.train_label_type,
            **model_args,
            **kwargs,
        )

    def transform_corpus(self, model, corpus):
        model.add_and_switch_to_new_task(
            task_name="2_CLASS",
            label_dictionary=corpus.make_label_dictionary(self.train_label_type),
            label_type=self.train_label_type,
        )
        return corpus

    @pytest.mark.integration()
    def test_predict_zero_shot(self, loaded_pretrained_model):
        sentence = Sentence("I am so glad you liked it!")
        loaded_pretrained_model.predict_zero_shot(sentence, ["happy", "sad"])
        assert len(sentence.get_labels(loaded_pretrained_model.label_type)) == 1
        assert sentence.get_labels(loaded_pretrained_model.label_type)[0].value == "happy"

    @pytest.mark.integration()
    def test_predict_zero_shot_single_label_always_predicts(self, loaded_pretrained_model):
        sentence = Sentence("I hate it")
        loaded_pretrained_model.predict_zero_shot(sentence, ["happy", "sad"])
        # Ensure this is an example that predicts no classes in multilabel
        assert len(sentence.get_labels(loaded_pretrained_model.label_type)) == 0
        loaded_pretrained_model.predict_zero_shot(sentence, ["happy", "sad"], multi_label=False)
        assert len(sentence.get_labels(loaded_pretrained_model.label_type)) == 1
        assert sentence.get_labels(loaded_pretrained_model.label_type)[0].value == "sad"

    @pytest.mark.integration()
    def test_init_tars_and_switch(self, tasks_base_path, corpus):
        tars = TARSClassifier(
            task_name="2_CLASS",
            label_dictionary=corpus.make_label_dictionary(label_type="class"),
            label_type="class",
        )

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 2

        # switch to task with only one label
        tars.add_and_switch_to_new_task("1_CLASS", "one class", "testlabel")

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 1

        # switch to task with three labels provided as list
        tars.add_and_switch_to_new_task("3_CLASS", ["list 1", "list 2", "list 3"], "testlabel")

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 3

        # switch to task with four labels provided as set
        tars.add_and_switch_to_new_task("4_CLASS", {"set 1", "set 2", "set 3", "set 4"}, "testlabel")

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 4

        # switch to task with two labels provided as Dictionary
        tars.add_and_switch_to_new_task("2_CLASS_AGAIN", corpus.make_label_dictionary(label_type="class"), "testlabel")

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 2

    @pytest.mark.skip("embeddings are not supported in tars")
    def test_load_use_model_keep_embedding(self):
        pass

    @pytest.mark.skip("tars needs additional setup after loading")
    def test_load_use_model(self):
        pass
