from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.models.text_classification_model import TARSClassifier


def test_init_tars_and_switch(tasks_base_path):

    # test corpus
    corpus = ClassificationCorpus(tasks_base_path / "imdb")

    # create a TARS classifier
    tars = TARSClassifier(task_name='2_CLASS', label_dictionary=corpus.make_label_dictionary())

    # check if right number of classes
    assert(len(tars.label_dictionary) == 2)

    # switch to task with only one label
    tars.add_and_switch_to_new_task('1_CLASS', 'one class')

    # check if right number of classes
    assert(len(tars.label_dictionary) == 1)

    # switch to task with three labels provided as list
    tars.add_and_switch_to_new_task('3_CLASS', ['list 1', 'list 2', 'list 3'])

    # check if right number of classes
    assert(len(tars.label_dictionary) == 3)

    # switch to task with four labels provided as set
    tars.add_and_switch_to_new_task('4_CLASS', {'set 1', 'set 2', 'set 3', 'set 4'})

    # check if right number of classes
    assert(len(tars.label_dictionary) == 4)

    # switch to task with two labels provided as Dictionary
    tars.add_and_switch_to_new_task('2_CLASS_AGAIN', corpus.make_label_dictionary())

    # check if right number of classes
    assert(len(tars.label_dictionary) == 2)