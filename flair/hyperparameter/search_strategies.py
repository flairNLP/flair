from abc import abstractmethod
import random
import logging
import numpy as np
from random import randrange

from .parameter_collections import ParameterStorage
from .search_spaces import SearchSpace

log = logging.getLogger("flair")


class SearchStrategy(object):
    """
    Parent object for all implemented search strategies. They define how the optimization procedure is.
    """

    def __init__(self):
        self.search_strategy_name = self.__class__.__name__

    @abstractmethod
    def make_configurations(self, parameter_storage: ParameterStorage):
        """
        has to be overwritten, makes configurations (single training instance) out of the storage object (parameter - parameter value pairs)
        :param parameter_storage: ParameterStorage Object
        """
        pass


class GridSearch(SearchStrategy):
    """
    Standard grid search procedure by making cartesian product from all possible configurations and then process
    them in order.
    """

    def __init__(self, shuffle: bool = False):
        """
        :param shuffle: bool. True if order of configuration objects should be shuffled.
        """
        super().__init__()
        self.shuffle = shuffle

    def make_configurations(self, search_space: SearchSpace):
        """
        override method. makes single training configurations from storage object.
        :param search_space: SearchSpace object
        :return: -
        """
        search_space.check_completeness(self.search_strategy_name)
        search_space.training_configurations.make_grid_configurations(search_space.parameter_storage,
                                                                      search_space.has_document_embeddings)
        if self.shuffle:
            random.shuffle(search_space.training_configurations.configurations)


class RandomSearch(GridSearch):
    """
    Same as GridSearch apart of training configurations are going to be shuffled.
    """

    def __init__(self):
        super().__init__(shuffle=True)


class EvolutionarySearch(SearchStrategy):

    def __init__(
            self,
            population_size: int = 12,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.05,
    ):
        """
        :param population_size: size of configurations per generation
        :param cross_rate: probability that configurations will pass their parameters to next generation
        :param mutation_rate: probability that parameters will be mutated randomly
        """
        super().__init__()
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def make_configurations(self, search_space: SearchSpace):
        """
        override method. makes single training configurations from storage object.
        :param search_space: SearchSpace object
        :return: -
        """
        search_space.check_completeness(self.search_strategy_name)
        search_space.training_configurations.make_evolutionary_configurations(search_space.parameter_storage,
                                                                              search_space.has_document_embeddings,
                                                                              self.population_size)

    def evolve_required(self, current_run: int) -> bool:
        """
        checks if evolve if required. Returns True if generation is over.
        :return: bool. True if generation is over
        """
        if current_run % self.population_size == 0:
            return True
        else:
            return False

    def evolve(self, search_space: SearchSpace, results: dict):
        """
        evolution of current generation. applies to each individual crossover() and mutate() according to set probabilities
        finally adds selected, crossovered and mutated training configurations to new generation.
        :param search_space: SearchSpace object
        :param results: dict containing results of current generation
        :return: -
        """
        current_results = self._get_current_results(results)
        log.info(50 * '-')
        log.info("Starting evolution - this generation had following results:")
        for idx, result in enumerate(current_results.values()):
            log.info(f"Configuration {idx} with a result of {result.get('result')}.")
        parent_population = self._get_parent_population(current_results)
        selected_population = self._select(current_results)
        for idx, child in enumerate(selected_population):
            log.info(50 * '-')
            log.info(f"Performing crossover and mutation for configuration {idx}.")
            child = self._crossover(child, parent_population)
            child = self._mutate(child, search_space.parameter_storage)
            search_space.training_configurations.add_configuration(child)
        log.info("Evolution completed.")

    def _get_current_results(self, results: dict):
        """
        extracts current results from dict
        :param results: dict containing the results
        :return: last X (size of generation) results
        """
        most_recent_ids = np.arange(0, len(results))[-self.population_size:]
        key_generator = lambda id: f"training-run-{id}"
        current_keys = list(map(key_generator, most_recent_ids))
        current_results = {key: value for key, value in results.items() if key in current_keys}
        return current_results

    def _get_parent_population(self, results: dict) -> dict:
        """
        extracts parent populaton from result dict (the generation which is just over)
        :param results: dict containing the results
        :return: grouped population by embedding keys for further processing
        """
        parent_population = self._extract_configurations_from_results(results)
        grouped_parent_population = self._group_by_embedding_keys(parent_population)
        return grouped_parent_population

    @staticmethod
    def _extract_configurations_from_results(results: dict) -> list:
        """
        extracts training configurations from results
        :param results: dict containing the current results
        :return: list of current configurations
        """
        configurations = []
        for configuration in results.values():
            configurations.append(configuration.get("params"))
        return configurations

    def _group_by_embedding_keys(self, parent_population: list) -> dict:
        """
        groups a list of training configuration by their embedding keys
        :param parent_population: list of training configurations
        :return: grouped list of training configurations by embedding key
        """
        grouped_parent_population = {}
        for embedding in parent_population:
            embedding_key = self._get_embedding_key(embedding)
            embedding_value = embedding
            if embedding_key in grouped_parent_population:
                grouped_parent_population[embedding_key].append(embedding_value)
            else:
                grouped_parent_population[embedding_key] = [embedding_value]
        return grouped_parent_population

    @staticmethod
    def _get_embedding_key(parameters: dict):
        """
        extracts embedding key from dict
        :param embedding: dict containing key value pairs of "parameter":"parameter_value"
        """
        if parameters.get("document_embeddings") is not None:
            embedding_key = parameters.get('document_embeddings').__name__
        else:
            embedding_key = "GeneralParameters"
        return embedding_key

    def _select(self, current_results: dict) -> np.array:
        """
        depending on its results, select the best fitting configurations based relative performance compared to the rest of the population
        For instance - Score of Run 1: 0.75, Run 2: 0.9, Run 3: 0.35 then selection probabilites are given by [0.375, 0.45, 0.175]
        :param current_results: dict containing current results
        :return: selected training configuration according to selection probabilites.
        """
        current_configurations = [result.get("params") for result in current_results.values()]
        evolution_probabilities = self._get_fitness(current_results)
        log.info(50 * '-')
        for idx, prob in enumerate(evolution_probabilities):
            log.info(f"The evolution probability for configuration {idx} is: {prob}.")
        return np.random.choice(current_configurations, size=self.population_size, replace=True,
                                p=evolution_probabilities)

    @staticmethod
    def _get_fitness(results: dict):
        """
        calculates the fitness by taking each training configuration's score. selection probabilities are then given
        by dividing the individual score with the total sum of scores.
        :param results: dict containing the results
        :return probabilities: np.array containing selection probabilities per individual in generation
        """
        fitness = np.asarray([configuration['result'] for configuration in results.values()])
        probabilities = fitness / (sum([configuration['result'] for configuration in results.values()]))
        return probabilities

    def _crossover(self, child: dict, parent_population: dict):
        """
        performs crossover between a parent configuration (from previous generation) and a selected configuration for the upcoming generation.
        selects randomly a matching training configuration from previous generation (parent) and then checks for all parameters whether the
        parameter configuration from parent is given to the new configuration according the the crossover probability set.
        :param child: dict containing the parameters of a child
        :param parent_population: dict containing all configurations from previous population
        :return: crossover child
        """
        child_type = self._get_embedding_key(child)
        configuration_with_same_embedding = len(parent_population[child_type])
        DNA_size = len(child)
        if np.random.rand() < self.cross_rate:
            random_configuration = randrange(configuration_with_same_embedding)  # select another individual from pop
            parent = {**parent_population.get(child_type)[random_configuration]}
            cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)  # choose crossover points
            for (parameter, value), replace in zip(child.items(), cross_points):
                if replace:
                    child[parameter] = parent[parameter]  # mating and produce one child
            log.info(f"Crossover performed for current for configuration ({np.sum(cross_points)} parameters changed).")
        else:
            log.info("No crossover for current configuration.")
        return child

    def _mutate(self, child: dict, parameter_storage: ParameterStorage):
        """
        performs mutation (randomly select a parameter value from the parameter storage) to a child according the mutation
        probability set.
        :param child: dict containing the current training configuration
        :param parameter_storage: ParameterStorage Object
        :return: mutated training configuration
        """
        child_type = self._get_embedding_key(child)
        mutation_points = np.where(np.random.rand(len(child)) < self.mutation_rate, 1, 0).astype(bool)
        for parameter, replace in zip(child.keys(), mutation_points):
            if replace:
                if parameter in getattr(parameter_storage, child_type):
                    child[parameter] = self._sample_parameter(child_type, parameter, parameter_storage)
                elif parameter in getattr(parameter_storage, "GeneralParameters"):
                    child[parameter] = self._sample_parameter("GeneralParameters", parameter, parameter_storage)
        log.info(f"{np.sum(mutation_points)} parameters have been mutated.")
        return child

    @staticmethod
    def _sample_parameter(key: str, parameter: str, parameter_storage: ParameterStorage):
        """
        Samples a parameter from the ParameterStorage
        :param key: str if embedding specific key or general
        :param parameter: str of parameter to sample from
        :param parameter_storage: Storage object containing the value range of all parameters
        :return: sampled parameter value from storage object
        """
        return random.sample(getattr(parameter_storage, key).get(parameter), 1).pop()
