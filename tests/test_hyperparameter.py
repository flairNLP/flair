import shutil

import pytest
import ray
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch.optim import SGD

from flair.embeddings import (
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    DocumentRNNEmbeddings,
)
from flair.hyperparameter import (
    SearchSpace,
    Parameter,
    SequenceTaggerParamSelector,
    TextClassifierParamSelector,
)
import flair.datasets
from flair.hyperparameter.tune import FlairTune
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


@pytest.mark.integration
def test_tune_classifier(results_base_path, tasks_base_path):

    # setup experiment class
    class TuneTextClassifier(FlairTune):
        def _setup(self, config):
            # A. set the corpus you are using
            self.corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")

            # B. define your embeddings and hyperparameters
            embeddings = DocumentRNNEmbeddings(
                [WordEmbeddings("glove")],
                hidden_size=config["hidden_size"] if "hidden_size" in config else 128,
                dropout=config["dropout"] if "dropout" in config else 0.5,
                locked_dropout=config["locked_dropout"]
                if "locked_dropout" in config
                else 0.0,
                word_dropout=config["word_dropout"]
                if "word_dropout" in config
                else 0.0,
            )

            # define model
            model = TextClassifier(
                document_embeddings=embeddings,
                label_dictionary=self.corpus.make_label_dictionary(),
            )

            # C. define training parameters
            self.trainer = ModelTrainer(
                model,
                self.corpus,
                learning_rate=config["initial_lr"] if "initial_lr" in config else 0.1,
                patience=config["patience"] if "patience" in config else 3,
                mini_batch_size=config["mini_batch_size"]
                if "mini_batch_size" in config
                else 8,
            )

            self.embeddings_storage_mode = "gpu"

    # 1. set the number of parameter combinations to try
    number_of_experiments: int = 2

    # 2. set the maximum number of epochs per experiment
    max_epochs: int = 2

    # 3. set how many experiments at the same time and what resources does each get
    max_concurrent: int = 2
    resources_per_experiment = {"cpu": 1, "gpu": 0.0}

    # 4. define your search space
    search = HyperOptSearch(
        {
            "hidden_size": hp.choice("hidden_size", [32, 64, 128, 256]),
            "dropout": hp.uniform("dropout", 0.00, 0.8),
            "locked_dropout": hp.uniform("locked_dropout", 0.0, 0.8),
            "word_dropout": hp.uniform("word_dropout", 0.0, 0.2),
            "initial_lr": hp.uniform("initial_lr", 0.0, 0.5),
            "patience": hp.choice("patience", [1, 2, 3]),
            "mini_batch_size": hp.choice("mini_batch_size", [2, 4, 8, 16, 32]),
        },
        max_concurrent=max_concurrent,
        reward_attr="mean_accuracy",
    )

    experiment_name = "experiment"

    # Define scheduler (optional)
    scheduler = ray.tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        grace_period=1,
        max_t=max_epochs,
    )

    # run experiment
    tune.run(
        TuneTextClassifier,
        local_dir=results_base_path,
        name=experiment_name,
        scheduler=scheduler,
        stop={"1-lr": 0.9999, "training_iteration": max_epochs},
        resources_per_trial=resources_per_experiment,
        num_samples=number_of_experiments,
        search_alg=search,
        return_trials=False,
    )

    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_sequence_tagger_param_selector(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(
        data_folder=tasks_base_path / "fashion", column_format={0: "text", 2: "ner"}
    )

    # define search space
    search_space = SearchSpace()

    # sequence tagger parameter
    search_space.add(
        Parameter.EMBEDDINGS,
        hp.choice,
        options=[
            StackedEmbeddings([WordEmbeddings("glove")]),
            StackedEmbeddings(
                [
                    WordEmbeddings("glove"),
                    FlairEmbeddings("news-forward-fast"),
                    FlairEmbeddings("news-backward-fast"),
                ]
            ),
        ],
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
    optimizer = SequenceTaggerParamSelector(
        corpus, "ner", results_base_path, max_epochs=2
    )
    optimizer.optimize(search_space, max_evals=2)

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_text_classifier_param_selector(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "imdb")

    glove_embedding: WordEmbeddings = WordEmbeddings("glove")

    search_space = SearchSpace()

    # document embeddings parameter
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[[glove_embedding]])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128, 256, 512])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.REPROJECT_WORDS, hp.choice, options=[True, False])
    search_space.add(Parameter.REPROJECT_WORD_DIMENSION, hp.choice, options=[64, 128])
    search_space.add(Parameter.BIDIRECTIONAL, hp.choice, options=[True, False])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.25, high=0.75)
    search_space.add(Parameter.WORD_DROPOUT, hp.uniform, low=0.25, high=0.75)
    search_space.add(Parameter.LOCKED_DROPOUT, hp.uniform, low=0.25, high=0.75)

    # training parameter
    search_space.add(Parameter.LEARNING_RATE, hp.uniform, low=0, high=1)
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4, 8, 16, 32])
    search_space.add(Parameter.ANNEAL_FACTOR, hp.uniform, low=0, high=0.75)
    search_space.add(Parameter.PATIENCE, hp.choice, options=[3, 5])

    param_selector = TextClassifierParamSelector(
        corpus, False, results_base_path, document_embedding_type="lstm", max_epochs=2
    )
    param_selector.optimize(search_space, max_evals=2)

    # clean up results directory
    shutil.rmtree(results_base_path)
