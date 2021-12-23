from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer


def test_init_tars_and_switch(tasks_base_path):
    # test corpus
    corpus = ClassificationCorpus(tasks_base_path / "imdb")

    # create a TARS classifier
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


def test_train_tars(tasks_base_path, results_base_path):
    # test corpus
    corpus = ClassificationCorpus(tasks_base_path / "imdb_underscore")

    # create a TARS classifier
    tars = TARSClassifier(embeddings="sshleifer/tiny-distilroberta-base")

    # switch to a new task (TARS can do multiple tasks so you must define one)
    tars.add_and_switch_to_new_task(
        task_name="question 2_CLASS",
        label_dictionary=corpus.make_label_dictionary(label_type="class"),
        label_type="class",
    )

    # initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # start the training
    trainer.train(
        base_path=results_base_path,
        learning_rate=0.02,
        mini_batch_size=1,
        max_epochs=1,
    )

    sentence = Sentence("This is great!")
    tars.predict(sentence)
