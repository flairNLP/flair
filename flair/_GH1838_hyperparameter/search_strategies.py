from abc import abstractmethod
import random
import logging
import numpy as np
from random import randrange

from FlairParamOptimizer.parameter_collections import ParameterStorage
from FlairParamOptimizer.search_spaces import SearchSpace

log = logging.getLogger("flair")

class SearchStrategy(object):

    def __init__(self):
        self.search_strategy_name = self.__class__.__name__

    @abstractmethod
    def make_configurations(self, parameter_storage: ParameterStorage):
        pass


class GridSearch(SearchStrategy):

    def __init__(self, shuffle : bool = False):
        super().__init__()
        self.shuffle = shuffle

    def make_configurations(self, search_space: SearchSpace):
        search_space.check_completeness(self.search_strategy_name)
        search_space.training_configurations.make_grid_configurations(search_space.parameter_storage)
        if self.shuffle:
            random.shuffle(search_space.training_configurations.configurations)


class RandomSearch(GridSearch):

    def __init__(self):
        super().__init__(shuffle=True)

class EvolutionarySearch(SearchStrategy):

    def __init__(
            self,
            population_size: int = 12,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.05,
    ):
        super().__init__()
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def make_configurations(self, search_space: SearchSpace):
        search_space.check_completeness(self.search_strategy_name)
        search_space.training_configurations.make_evolutionary_configurations(search_space.parameter_storage, self.population_size)

    def _evolve_required(self, current_run: int):
        if current_run % (self.population_size) == 0:
            return True
        else:
            return False

    def _evolve(self, search_space: SearchSpace, results: dict):
        current_results = self._get_current_results(results)
        log.info(50*'-')
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
            search_space.training_configurations._add_configuration(child)
        log.info("Evolution completed.")

    def _get_current_results(self, results:dict):
        most_recent_ids = np.arange(0, len(results))[-self.population_size:]
        key_generator = lambda id: f"training-run-{id}"
        current_keys = list(map(key_generator, most_recent_ids))
        current_results = {key: value for key, value in results.items() if key in current_keys}
        return current_results

    def _get_parent_population(self, results: dict) -> dict:
        parent_population = self._extract_configurations_from_results(results)
        grouped_parent_population = self._group_by_embedding_keys(parent_population)
        return grouped_parent_population

    def _extract_configurations_from_results(self, results: dict) -> list:
        configurations = []
        for configuration in results.values():
            configurations.append(configuration.get("params"))
        return configurations

    def _group_by_embedding_keys(self, parent_population: list) -> dict:
        grouped_parent_population = {}
        for embedding in parent_population:
            embedding_key = self._get_embedding_key(embedding)
            embedding_value = embedding
            if embedding_key in grouped_parent_population:
                grouped_parent_population[embedding_key].append(embedding_value)
            else:
                grouped_parent_population[embedding_key] = [embedding_value]
        return grouped_parent_population

    def _get_embedding_key(self, embedding: dict):
        if embedding.get("document_embeddings") is not None:
            embedding_key = embedding.get('document_embeddings').__name__
        else:
            embedding_key = "GeneralParameters"
        return embedding_key

    def _select(self, current_results: dict) -> np.array:
        current_configurations = [result.get("params") for result in current_results.values()]
        evolution_probabilities = self._get_fitness(current_results)
        log.info(50 * '-')
        for idx, prob in enumerate(evolution_probabilities):
            log.info(f"The evolution probability for configuration {idx} is: {prob}.")
        return np.random.choice(current_configurations, size=self.population_size, replace=True, p=evolution_probabilities)

    def _get_fitness(self, results: dict):
        fitness = np.asarray([configuration['result'] for configuration in results.values()])
        probabilities = fitness / (sum([configuration['result'] for configuration in results.values()]))
        return probabilities

    def _crossover(self, child: dict, parent_population: dict):
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

    def _sample_parameter(self, key: str, parameter: str, parameter_storage: ParameterStorage):
        return random.sample(getattr(parameter_storage, key).get(parameter), 1).pop()