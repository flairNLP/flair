import shutil

from flair.data import Dictionary
from flair.trainers.language_model_trainer import TextCorpus


def test_train_resume_language_model_training(resources_path, results_base_path, tasks_base_path):
    # get default dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # get the example corpus and process at character level in forward direction
    corpus: TextCorpus = TextCorpus(resources_path / 'corpora/lorem_ipsum',
                                dictionary,
                                forward=True,
                                character_level=True)

    assert (corpus.test is not None)
    assert (corpus.train_files is not None)
    assert (corpus.valid is not None)
    assert (len(corpus.train_files) == 2)
