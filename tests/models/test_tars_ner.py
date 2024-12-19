import pytest

import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TARSTagger
from tests.model_test_utils import BaseModelTest


class TestTarsTagger(BaseModelTest):
    model_cls = TARSTagger
    train_label_type = "ner"
    model_args = {"task_name": "2_NER"}
    training_args = {"mini_batch_size": 1, "max_epochs": 2}
    # pretrained_model = "tars-ner"  # disabled due to too much space requirements.

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})

    @pytest.fixture()
    def embeddings(self):
        return TransformerWordEmbeddings("distilbert-base-uncased")

    @pytest.fixture()
    def example_sentence(self):
        return Sentence("George Washington was born in Washington")

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
            task_name="2_NER",
            label_dictionary=corpus.make_label_dictionary(self.train_label_type),
            label_type=self.train_label_type,
        )
        return corpus

    @pytest.mark.integration()
    def test_predict_zero_shot(self, loaded_pretrained_model):
        sentence = Sentence("George Washington was born in Washington")
        loaded_pretrained_model.predict_zero_shot(sentence, ["location", "person"])
        assert len(sentence.get_labels("location-person")) == 2
        assert sorted([label.value for label in sentence.get_labels("location-person")]) == [
            "location",
            "person",
        ]

    @pytest.mark.integration()
    def test_init_tars_and_switch(self, tasks_base_path, corpus):
        tars = TARSTagger(
            task_name="2_NER",
            label_dictionary=corpus.make_label_dictionary(label_type="ner"),
            label_type="ner",
        )

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 10

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
        tars.add_and_switch_to_new_task("2_CLASS_AGAIN", corpus.make_label_dictionary(label_type="ner"), "testlabel")

        # check if right number of classes
        assert len(tars.get_current_label_dictionary()) == 10

    @pytest.mark.skip("embeddings are not supported in tars")
    def test_load_use_model_keep_embedding(self):
        pass

    @pytest.mark.skip("tars needs additional setup after loading")
    def test_load_use_model(self):
        pass
