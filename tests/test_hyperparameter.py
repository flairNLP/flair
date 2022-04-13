import shutil

import pytest
from hyperopt import hp
from torch.optim import SGD

import flair.datasets
from flair.embeddings import StackedEmbeddings, WordEmbeddings
from flair.hyperparameter import (
    Parameter,
    SearchSpace,
    SequenceTaggerParamSelector,
    TextClassifierParamSelector,
)

glove_embedding: WordEmbeddings = WordEmbeddings("glove")


@pytest.mark.integration
def test_sequence_tagger_param_selector(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})

    # define search space
    search_space = SearchSpace()

    # sequence tagger parameter
    search_space.add(
        Parameter.EMBEDDINGS,
        hp.choice,
        options=[StackedEmbeddings([glove_embedding])],
    )
    search_space.add(Parameter.USE_CRF, hp.choice, options=[True, False])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.25, high=0.75)
    search_space.add(Parameter.WORD_DROPOUT, hp.uniform, low=0.0, high=0.25)
    search_space.add(Parameter.LOCKED_DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])

    # model trainer parameter
    search_space.add(Parameter.OPTIMIZER, hp.choice, options=[SGD])

    # training parameter
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4, 8, 32])
    search_space.add(Parameter.LEARNING_RATE, hp.uniform, low=0.01, high=1)
    search_space.add(Parameter.ANNEAL_FACTOR, hp.uniform, low=0.3, high=0.75)
    search_space.add(Parameter.PATIENCE, hp.choice, options=[3, 5])
    search_space.add(Parameter.WEIGHT_DECAY, hp.uniform, low=0.01, high=1)

    # find best parameter settings
    optimizer = SequenceTaggerParamSelector(corpus, "ner", results_base_path, max_epochs=2)
    optimizer.optimize(search_space, max_evals=2)

    # clean up results directory
    shutil.rmtree(results_base_path)
    del optimizer, search_space


@pytest.mark.integration
def test_text_classifier_param_selector(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb", label_type="sentiment")
    label_type = "sentiment"

    search_space = SearchSpace()

    # document embeddings parameter
    search_space.add(Parameter.TRANSFORMER_MODEL, hp.choice, options=["sshleifer/tiny-distilbert-base-cased"])
    search_space.add(Parameter.LAYERS, hp.choice, options=["-1", "-2"])

    # training parameter
    search_space.add(Parameter.LEARNING_RATE, hp.uniform, low=0, high=1)
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4, 8, 16, 32])
    search_space.add(Parameter.ANNEAL_FACTOR, hp.uniform, low=0, high=0.75)
    search_space.add(Parameter.PATIENCE, hp.choice, options=[3, 5])

    param_selector = TextClassifierParamSelector(corpus, label_type, False, results_base_path, max_epochs=2)
    param_selector.optimize(search_space, max_evals=2)

    # clean up results directory
    shutil.rmtree(results_base_path)
    del param_selector, search_space
