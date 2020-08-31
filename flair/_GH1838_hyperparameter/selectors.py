import time
import os
import logging
import pickle
from datetime import datetime
from operator import getitem
from typing import Union
from pathlib import Path
from torch import cuda
from abc import abstractmethod

from GeneticParamOptimizer.hyperparameter.optimizers import *
from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace
from GeneticParamOptimizer.hyperparameter.helpers import *

import flair.nn
from flair.data import Corpus
from flair.datasets import *
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import TextClassifier

log = logging.getLogger("flair")

class ParamSelector():
    """
    The ParamSelector selects the best configuration omitted by an optimizer object
    Attributes:
        corpus: the downstream task corpus
        base_path: path where to store results
        optimizer: Optimizer object
        search_space: SearchSpace object
    """

    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer: ParamOptimizer,
            search_space: SearchSpace,
    ):

        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus = corpus
        self.base_path = base_path
        self.optimizer = optimizer
        self.search_space = search_space
        self.current_run = 0

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        """
        Sets up the respective downstream task model
        :param params: dict of parameters for model configuration
        :return:
        """
        pass

    @abstractmethod
    def _train(self, params: dict):
        """
        trains the respective downstream task model
        :param params: dict of parameters for training configuration
        :return:
        """
        pass

    def optimize(self, train_on_multiple_gpus : bool = False):
        """
        optimize over the search space
        :param train_on_multiple_gpus: if true, use PyTorch Distributed Data Parallel, requires at least two GPUs
        :return: None
        """
        while self._budget_is_not_used_up():

            current_configuration = self._get_current_configuration()

            if train_on_multiple_gpus and self._sufficient_available_gpus():
                self._perform_training_on_multiple_gpus(current_configuration)
            else:
                self._perform_training(current_configuration)

            if self.optimizer.__class__.__name__ == "GeneticOptimizer" \
            and self.optimizer._evolve_required(current_run=self.current_run):
                self.optimizer._evolve()

            self.current_run += 1

        self._log_results()

    def _perform_training(self, params: dict):
        """
        perfoms sequentiell training and stores result in optimizer.results
        :param params: dict containing the parameter configuration
        :return:
        """
        self.optimizer.results[f"training-run-{self.current_run}"] = self._train(params)
        self._store_results(result=self.optimizer.results[f"training-run-{self.current_run}"], current_run=self.current_run)

    def _perform_training_on_multiple_gpus(self, params: dict):
        #TODO to be implemented
        pass

    def _budget_is_not_used_up(self):
        """
        wrapper function to check whether budget is used up and to stop optimization
        :return:
        """

        budget_type = self._get_budget_type(self.search_space.budget)

        if budget_type == 'time_in_h':
            return self._is_time_budget_left()
        elif budget_type == 'runs':
            return self._is_runs_budget_left()
        elif budget_type == 'generations':
            return self._is_generations_budget_left()

    def _get_budget_type(self, budget: dict):
        """
        returns budget type
        :param budget: dict containing budget information
        :return:
        """
        if len(budget) == 1:
            for budget_type in budget.keys():
                return budget_type
        else:
            raise Exception('Budget has more than 1 parameter.')

    def _is_time_budget_left(self):
        """
        checks whether time budget is not exceeded
        :return: True if time is left
        """
        already_running = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.search_space.start_time)
        if (already_running.total_seconds()) / 3600 < self.search_space.budget['time_in_h']:
            return True
        else:
            return False

    def _is_runs_budget_left(self):
        """
        checks whether runs budget is left. If yes decrease runs by 1.
        :return: True if runs are left
        """
        if self.search_space.budget['runs'] > 0:
            self.search_space.budget['runs'] -= 1
            return True
        else:
            return False

    def _is_generations_budget_left(self):
        """
        checks whether generations budget is left.
        Generations gets decreased if left generations are greater than 1 and every X steps where X is the size of the population per generation.
        :return: True if budget is left
        """
        if self.search_space.budget['generations'] > 1 \
        and self.current_run % self.optimizer.population_size == 0\
        and self.current_run != 0:
            self.search_space.budget['generations'] -= 1
            return True

        elif self.search_space.budget['generations'] == 1 \
        and self.current_run % self.optimizer.population_size == 0\
        and self.current_run != 0:
            self.search_space.budget['generations'] -= 1
            return False

        elif self.search_space.budget['generations'] > 0:
            return True

        else:
            return False

    def _get_current_configuration(self):
        """
        return current configuration from optimizer.configurations
        :return: dict of parameters
        """
        current_configuration = self.optimizer.configurations[self.current_run]
        return current_configuration

    def _sufficient_available_gpus(self):
        """
        If training is set to multiple GPUs, checks whether enough GPUs are available
        :return: True if more than 1 GPU is available
        """
        if cuda.device_count() > 1:
            return True
        else:
            log.info("There are less than 2 GPUs available, switching to standard calculation.")

    def _log_results(self):
        """
        When budget is used up, log best results
        :return: None
        """
        sorted_results = sorted(self.optimizer.results.items(), key=lambda x: getitem(x[1], 'result'), reverse=True)[:5]
        log.info("The top 5 results are:")
        for idx, config in enumerate(sorted_results):
            log.info(50*'-')
            log.info(idx+1)
            log.info(f"{config[0]} with a score of {config[1]['result']}.")
            log.info("with following configurations:")
            for parameter, value in config[1]['params'].items():
                log.info(f"{parameter}:  {value}")

    def _store_results(self, result: dict, current_run: int):
        """
        stores a .txt file with the results
        :return: None
        """
        result['timestamp'] = datetime.now()
        entry = f"training-run-{current_run}"
        try:
            self._load_and_pickle_results(entry, result)
        except FileNotFoundError:
            self._initialize_results_pickle(entry, result)

    def _load_and_pickle_results(self, entry: str, result: dict):
        pickle_file = open(self.base_path / "results.pkl", 'rb')
        results_dict = pickle.load(pickle_file)
        pickle_file.close()
        pickle_file = open(self.base_path / "results.pkl", 'wb')
        results_dict[entry] = result
        pickle.dump(results_dict, pickle_file)
        pickle_file.close()

    def _initialize_results_pickle(self, entry: str, result: dict):
        results_dict = {}
        pickle_file = open(self.base_path / "results.pkl", 'wb')
        results_dict[entry] = result
        pickle.dump(results_dict, pickle_file)
        pickle_file.close()





class TextClassificationParamSelector(ParamSelector):
    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            optimizer: ParamOptimizer,
            search_space: SearchSpace,
            multi_label: bool = False,
    ):
        super().__init__(
            corpus,
            base_path,
            optimizer=optimizer,
            search_space=search_space,
        )

        self.multi_label = multi_label

    def _set_up_model(self, params: dict, label_dictionary : dict):
        """
        Creates an text classifier object with the respective document embeddings.
        :param params: dict of parameters
        :param label_dictionary: label dictionary
        :return: TextClassifier object
        """

        document_embedding = params['document_embeddings'].__name__
        if document_embedding == "DocumentRNNEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_RNN_EMBEDDING_PARAMETERS
            }
            embedding_params['embeddings'] = [WordEmbeddings(TokenEmbedding) if type(params['embeddings']) == list
                                              else WordEmbeddings(params['embeddings']) for TokenEmbedding in params['embeddings']]
            document_embedding = DocumentRNNEmbeddings(**embedding_params)

        elif document_embedding == "DocumentPoolEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }
            embedding_params['embeddings'] = [WordEmbeddings(TokenEmbedding) for TokenEmbedding in params['embeddings']]
            document_embedding = DocumentPoolEmbeddings(**embedding_params)

        elif document_embedding == "TransformerDocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS
            }
            document_embedding = TransformerDocumentEmbeddings(**embedding_params)

        else:
            raise Exception("Please provide a flair document embedding class")

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=document_embedding,
        )

        return text_classifier

    def _train(self, params: dict):
        """
        trains a TextClassifier with given configuration
        :param params: dict containing the parameter
        :return: dict containing result and configuration
        """

        corpus = self.corpus

        label_dict = corpus.make_label_dictionary()

        for sent in corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params, label_dict)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
        }

        trainer: ModelTrainer = ModelTrainer(
            model, corpus, **model_trainer_parameters
        )

        path = Path(self.base_path) / f"training-run-{self.current_run}"

        embeddings_storage_mode = 'gpu' if cuda.is_available() else 'cpu'

        results = trainer.train(
            path,
            max_epochs=self.search_space.max_epochs_per_training,
            param_selection_mode=True,
            **training_params,
            embeddings_storage_mode=embeddings_storage_mode,
        )

        if self.search_space.optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}

class SequenceTaggerParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        base_path: Union[str, Path],
        optimizer: Optimizer,
        search_space: SearchSpace,
    ):
        """
        :param corpus: the corpus
        :param tag_type: tag type to use
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        """
        super().__init__(
            corpus,
            base_path,
            optimizer=optimizer,
            search_space=search_space
        )

        self.tag_type = search_space.tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        """
        sets up the sequence tagger object for a given configuration
        :param params: dict containing the parameters
        :return: SequenceTagger object
        """

        sequence_tagger_params = {
            key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS
        }

        embedding_types = params['embeddings']

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        sequence_tagger_params['embeddings'] = embeddings

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )

        return tagger

    def _train(self, params: dict):
        """
        trains a sequence tagger model
        :param params: dict containing the parameters
        :return: dict containing result and configuration
        """

        corpus = self.corpus

        tagger = self._set_up_model(params=params)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
        }

        trainer: ModelTrainer = ModelTrainer(
            tagger, corpus, **model_trainer_parameters
        )

        path = Path(self.base_path) / f"training-run-{self.current_run}"

        embeddings_storage_mode = 'gpu' if cuda.is_available() else 'cpu'

        results = trainer.train(path,
                      max_epochs=self.search_space.max_epochs_per_training,
                      **training_params,
                      embeddings_storage_mode=embeddings_storage_mode)

        if self.search_space.optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}