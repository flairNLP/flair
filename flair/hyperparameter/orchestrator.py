import pickle
from datetime import datetime
from operator import getitem
from typing import Union
from pathlib import Path
from torch import cuda

from .search_strategies import *
from.search_spaces import SearchSpace
from .downstream_task_models import TextClassification, SequenceTagging

from flair.data import Corpus

log = logging.getLogger("flair")

class Orchestrator(object):

    def __init__(
            self,
            corpus: Corpus,
            base_path: Union[str, Path],
            search_space: SearchSpace,
            search_strategy: SearchStrategy
    ):
        if type(base_path) is str:
            base_path = Path(base_path)
        self.corpus = corpus
        self.base_path = base_path
        self.search_space = search_space
        self.search_strategy = search_strategy
        self.downstream_task_model = self._get_downstream_task_model_from_class_name(search_space.__class__.__name__)
        self.current_run = 0
        self.results = {}

        # This is required because we'll calculate generations budget with modulo operator (decrease budget every X configurations)
        if search_strategy.search_strategy_name == "EvolutionarySearch":
            search_space.budget._set_population_size(search_strategy.population_size)

    def _get_downstream_task_model_from_class_name(self, downstream_task):
        if downstream_task == "TextClassifierSearchSpace":
            model = TextClassification(self.search_space.multi_label)
        elif downstream_task == "SequenceTaggerSearchSpace":
            model = SequenceTagging(self.search_space.tag_type)
        else:
            raise Exception("No known downstream task provided.")
        return model

    def optimize(self, train_on_multiple_gpus : bool = False):
        while self.search_space.budget._is_not_used_up() and self.search_space.training_configurations.has_configurations_left():
            current_configuration = self.search_space.training_configurations.get_configuration()
            if train_on_multiple_gpus and self._sufficient_available_gpus():
                self._perform_training_on_multiple_gpus(current_configuration)
            else:
                self._perform_training(current_configuration)
            if self.search_strategy.search_strategy_name == "EvolutionarySearch":
                if self.search_strategy._evolve_required(current_run=len(self.results)):
                    self.search_strategy._evolve(self.search_space, self.results)
        self._log_results()

    def _perform_training(self, params: dict):
        current_run = len(self.results)
        training_run_number = f"training-run-{current_run}"
        base_path = self.base_path / training_run_number
        try:
            self.results[training_run_number] = self.downstream_task_model._train(corpus=self.corpus,
                                                                                  params=params,
                                                                                  base_path= base_path,
                                                                                  max_epochs=self.search_space.max_epochs_per_training_run,
                                                                                  optimization_value=self.search_space.optimization_value)
        except RuntimeError:
            self.results[training_run_number] = {'result': 0, 'params': params}
        self._store_results(result=self.results[training_run_number], current_run=current_run)

    def _sufficient_available_gpus(self):
        if cuda.device_count() > 1:
            return True
        else:
            log.info("There are less than 2 GPUs available, switching to standard calculation.")

    def _perform_training_on_multiple_gpus(self, params: dict):
        #TODO to be implemented
        pass

    def _log_results(self):
        sorted_results = sorted(self.results.items(), key=lambda x: getitem(x[1], 'result'), reverse=True)[:5]
        log.info("The top 5 results are:")
        for idx, config in enumerate(sorted_results):
            log.info(50*'-')
            log.info(idx+1)
            log.info(f"{config[0]} with a score of {config[1]['result']}.")
            log.info("with following configurations:")
            for parameter, value in config[1]['params'].items():
                log.info(f"{parameter}:  {value}")

    def _store_results(self, result: dict, current_run: int):
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