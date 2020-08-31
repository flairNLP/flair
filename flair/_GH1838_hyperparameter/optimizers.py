import itertools
from abc import abstractmethod
import random
import logging as log
import numpy as np
from random import randrange
from .parameters import *

from GeneticParamOptimizer.hyperparameter.search_spaces import SearchSpace

"""
The ParamOptimizer object acts as the optimizer instance in flair's hyperparameter optimization.
We are currently supporting three types of optimization:
    GeneticOptimizer        using evolutionary algorithms
    GridSearchOptimizer     standard grid search optimization
    RandomSearchOptimizer   random order grid search optimization
    
The optimizers take an search space object as input, please see the documentation of search spaces for further
information.

Depending on the optimizer type the respective configurations of your hyperparameter optimization will be calculated.

Apart from optimizer specific functions, if you want to add a new optimization procedure, following functions
have to be overwritten:
    _get_configurations()                           wrapper function which returns a list of all configurations.
                                                    depending on whether embedding specific parameters are set,
                                                    call respective functions.
    _get_embedding_specific_configurations()        If we have embedding specific parameters.
    _get_standard_configurations()                  If we don't have embedding specific parameters.
    _get_configurations_for_single_embedding()      returns a list of configurations for one embedding.
"""

class ParamOptimizer(object):
    """
    Parent class for all hyperparameter optimizers.

    Attributes:
        search_space: Search Space object
    """

    def __init__(
            self,
            search_space: SearchSpace,
    ):
        """
        Parent class for optimizers which stores budget and parameters from search space object

        :rtype: object
        :param search_space: the search space from which to get parameters and budget from
        """
        self.results = {}
        self.document_embedding_specific = search_space.document_embedding_specific
        search_space._check_mandatory_parameters_are_set(optimizer_type=self.__class__.__name__)

    @abstractmethod
    #Wrapper function to get configurations
    def _get_configurations(self):
        pass

    @abstractmethod
    # If there are document embedding specific parameters
    def _get_embedding_specific_configurations(self):
        pass

    @abstractmethod
    # For standard parameters without document embeddings
    def _get_standard_configurations(self):
        pass

    @abstractmethod
    # returns all configurations for one embedding (either 1 document embedding or all parameter)
    def _get_configurations_for_single_embedding(self):
        pass


class GridSearchOptimizer(ParamOptimizer):
    """A class for grid search hyperparameter optimization."""

    def __init__(
            self,
            search_space: SearchSpace,
            shuffle: bool = False,
    ):
        """
        Creates a grid search object with all possible configurations from search space (cartesian product)

        :rtype: object
        :param search_space: the search space from which to get parameters and budget from
        :param shuffled: if true, returns a shuffled list of parameter configuration.
        """
        super().__init__(
            search_space
        )

        self.configurations = self._get_configurations(
                                parameters=search_space.parameters,
                                shuffle=shuffle,
                                embedding_specific=search_space.document_embedding_specific)

    def _get_configurations(
            self,
            parameters : dict,
            shuffle : bool,
            embedding_specific: bool
    ):
        """
        Wrapper function which does the cartesian product of provided configurations depending on search space type

        :param shuffled: if true, a shuffled list of configurations is returned
        :param parameters: a dict which contains parameters as keywords with its possible configurations as values
        :return: a list of parameters configuration
        :rtype: list of all configurations
        """

        if embedding_specific:
            configurations = self._get_embedding_specific_configurations(parameters, shuffle)
        else:
            configurations = self._get_standard_configurations(parameters, shuffle)

        return configurations

    def _get_embedding_specific_configurations(self,
                                     parameters: dict,
                                     shuffle: bool):
        """
        Returns all configurations and check embedding specific parameters,
        i.e. for the text classification downstream task
        :param parameters: Dict containing all parameters as key value pairs
        :param shuffled: Bool - if true, shuffle the grid
        :return: list of all configurations
        """

        all_configurations = []
        for document_embedding, embedding_parameters in parameters.items():
            all_configurations.append(self._get_configurations_for_single_embedding(embedding_parameters))

        all_configurations = self._flatten_grid(all_configurations)

        if shuffle:
            random.shuffle(all_configurations)

        return all_configurations

    def _get_standard_configurations(self, parameters: dict, shuffle: bool):
        """
        Returns all configurations for the sequence labeling downstream task
        :param parameters: Dict containing all parameters as key value pairs
        :param shuffled: Bool - if true, shuffle the grid
        :return: list of all configurations
        """

        all_configurations = self._get_configurations_for_single_embedding(parameters)

        if shuffle:
            random.shuffle(all_configurations)

        return all_configurations

    def _get_configurations_for_single_embedding(self, parameters: dict):
        """
        Returns the cartesian product for all configurations provided. Adds uniformly sampled data in the second step.
        :param parameters:
        :return:
        """

        option_parameters, uniformly_sampled_parameters = self._split_up_configurations(parameters)

        all_configurations = self._get_cartesian_product(option_parameters)

        # Since dicts are not sorted, uniformly sampled configurations have to be added later
        if uniformly_sampled_parameters:
            all_configurations = self._add_uniformly_sampled_parameters(uniformly_sampled_parameters, all_configurations)

        return all_configurations

    def _split_up_configurations(self, parameters: dict):
        """
        Splits the parameters based on whether to choose from options or take a uniform sample from a distribution.
        :param parameters: Dict containing the parameters
        :return: parameters from options as tuple, uniformly sampled parameters as dict
        """
        parameter_options = []
        parameter_keys = []
        uniformly_sampled_parameters = {}

        #TODO refactor with get operation
        for parameter_name, configuration in parameters.items():
            try:
                parameter_options.append(configuration['options'])
                parameter_keys.append(parameter_name)
            except:
                uniformly_sampled_parameters[parameter_name] = configuration

        return (parameter_keys, parameter_options), uniformly_sampled_parameters

    def _get_cartesian_product(self, parameters: tuple):
        """
        Returns the cartesian product of provided parameters. Takes two list (keys, values) in form of a tuple
        as input.
        :param parameters: tuple (list, list) containing keys and values of parameters
        :return: list of all configurations
        """
        parameter_keys, parameter_options = parameters
        all_configurations = []
        for configuration in itertools.product(*parameter_options):
            all_configurations.append(dict(zip(parameter_keys, configuration)))

        return all_configurations

    def _add_uniformly_sampled_parameters(self, bounds: dict, all_configurations: list):
        """
        Adds to each configuration a uniform sample of respective parameters.
        :param bounds: dict containing the parameters which should be uniformly sampled
        :param all_configurations: list of all configurations to which append a uniform sample parameter
        :return: list of all configurations with a uniformly sampled parameter
        """
        for item in all_configurations:
            for parameter_name, configuration in bounds.items():
                func = configuration['method']
                item[parameter_name] = func(configuration['bounds'])

        return all_configurations

    def _flatten_grid(self, all_configurations: list):
        """
        Flattens the list of all configurations for further processing.
        :param all_configurations: list of all configurations
        :return: flat list of all configurations
        """
        return [item for config in all_configurations for item in config]

class RandomSearchOptimizer(GridSearchOptimizer):
    """A class for random search hyperparameter optimization"""

    def __init__(
            self,
            search_space: SearchSpace,
    ):
        """
        Initializes a RandomSearchOptimizer object

        :param search_space: the search space from which to get parameters and budget from
        :rtype: object
        """
        super().__init__(
            search_space,
            shuffle=True
        )


class GeneticOptimizer(ParamOptimizer):
    """A class for hyperparameter optimization using evolutionary algorithms."""

    def __init__(
            self,
            search_space: SearchSpace,
            population_size: int = 8,
            cross_rate: float = 0.4,
            mutation_rate: float = 0.01,
    ):
        """
        Initializes a GeneticOptimizer for hyperparameter optimization using evolutionary algorithms

        :param search_space: the search space from which to get parameters and budget from
        :param population_size: number of configurations per generation
        :param cross_rate: percentage of crossover during recombination of configurations
        :param mutation_rate: probability of mutation of configurations
        :rtype: object
        """
        super().__init__(
            search_space
        )

        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.all_configurations = search_space.parameters

        self.configurations = self._get_configurations(
                                parameters=search_space.parameters,
                                embedding_specific=search_space.document_embedding_specific)

    def _get_configurations(
            self,
            parameters : dict,
            embedding_specific: str,
    ):
        """
        returns a generation of parameter configurations
        :param parameters: A dict containing all parameters
        :param population_size: size of population per generation
        :param embedding_specific: If true, document embedding specific parameters are provided (based on search space)
        :return: A list of all configurations
        """
        if embedding_specific:
            configurations = self._get_embedding_specific_configurations(parameters)
        else:
            configurations = self._get_standard_configurations(parameters)

        return configurations

    def _get_embedding_specific_configurations(self, parameters: dict):
        """
        Get all configurations for multiple (document) embedding type
        :param parameters: A dict containing parameters for a single embedding
        :param population_size: Size of population per generation
        :return: A list of parameter configurations
        """

        amount_individuals_per_embedding = self._get_amount_of_individuals_per_embedding(len(parameters))

        configurations = self._get_individuals_for_each_embedding(parameters, amount_individuals_per_embedding)

        random.shuffle(configurations)

        return configurations

    def _get_standard_configurations(self, parameters: dict):
        """
        Get all configurations for single (document) embedding type
        :param parameters: A dict containing parameters for a single embedding
        :param population_size: Size of population per generation
        :return: A list of parameter configurations
        """

        return self._get_configurations_per_single_embedding(parameters, self.population_size)

    def _get_configurations_per_single_embedding(self, parameters: dict, population_size: int):
        """
        Returns all configurations for one (document) embedding type
        :param parameters: A dict containing parameters for a single embedding
        :param population_size: Size of population per generation
        :return: A list of parameter configurations
        """
        individuals = []
        for idx in range(population_size):
            individual = {}
            for parameter_name, configuration in parameters.items():
                parameter_value = self._get_formatted_parameter_from(**configuration)
                individual[parameter_name] = parameter_value
            individuals.append(individual)

        return individuals

    def _get_amount_of_individuals_per_embedding(self, length_of_different_embeddings: int) -> list:
        """
        If multiple document embeddings, split initial population equally among all embedding types
        :param length_of_different_embeddings:
        :return: list of integers containing information about configurations per embedding
        """
        individuals_per_embedding = [self.population_size // length_of_different_embeddings +
                                  (1 if x < self.population_size % length_of_different_embeddings else 0)
                                  for x in range (length_of_different_embeddings)]

        return individuals_per_embedding

    def _get_individuals_for_each_embedding(self, parameters: dict, individuals_per_embedding: list):
        """
        Returns according to the embedding split, a list of parameter configurations
        :param parameters: a dict containing the parameter value
        :param individuals_per_embedding: list of ints how many individuals per embedding
        :return:
        """
        configurations = []

        for (nested_key, nested_parameters), individuals_per_group in zip(parameters.items(),
                                                                          individuals_per_embedding):
            configurations.append(
                self._get_configurations_per_single_embedding(nested_parameters, population_size=individuals_per_group))

        configurations = [item for embedding_type in configurations for item in embedding_type]

        return configurations

    def _get_formatted_parameter_from(self, **kwargs):
        """
        Helper function to extract parameter value depending on provided function

        :param kwargs: a tuple of a function and values / bounds
        :return: float or int depending on function provided
        """
        func = kwargs.get('method')
        if kwargs.get('options') is not None:
            parameter = func(kwargs.get('options'))
        elif kwargs.get('bounds') is not None:
            parameter = func(kwargs.get('bounds'))
        else:
            raise Exception("Please provide either bounds or options as arguments to the search space depending on your function.")
        return parameter

    def _evolve_required(self, current_run: int):
        """
        Checks if population has to be evolved
        :param current_run: int
        :return: True if all individuals from current population has been processed
        """
        if current_run % (self.population_size) == (self.population_size - 1):
            return True
        else:
            return False


    def _evolve(self):
        """
        Evolve the current population based on selection, mutation and crossover
        :param current_population: list contraining parameter configurations of current population
        :return: List of configuration (next generation)
        """
        parent_population = self._get_formatted_population()
        selected_population = self._select()
        for child in selected_population:
            child = self._crossover(child, parent_population)
            child = self._mutate(child)
            self.configurations.append(child)


    def _get_formatted_population(self):
        """
        Puts the input list in a processable format
        :param current_population: List of configurations
        :return: Formatted list of configuration
        """
        formatted = {}
        for embedding in self.configurations[-self.population_size:]:
            embedding_key = self._get_embedding_key(embedding)
            embedding_value = embedding
            if embedding_key in formatted:
                formatted[embedding_key].append(embedding_value)
            else:
                formatted[embedding_key] = [embedding_value]
        return formatted


    def _select(self):
        """
        Selects best fitting parameter configurations
        :param current_population: List of current configurations / population
        :return: List of best fitting individuals from current population
        """
        evo_probabilities = self._get_fitness()
        return np.random.choice(self.configurations, size=self.population_size, replace=True, p=evo_probabilities)


    def _get_fitness(self):
        """
        Calculates the fitness of each individual (individual fitness / sum of all fitnesses)
        :param current_population: list of all configurations
        :return: survival probabilities for each individual
        """
        fitness = np.asarray([individual['result'] for individual in self.results.values()])
        probabilities = fitness / (sum([individual['result'] for individual in self.results.values()]))
        return probabilities


    def _crossover(self, child: dict, parent_population: dict):
        """
        Given the crossover rate, randomly select a individual (parent) from current population and crossover parameters
        with child for next generation
        :param child: dict for a single training configuration
        :param parent_population: all selected individuals from previous generation
        :return: child with crossover parameters
        """
        child_type = self._get_embedding_key(child)
        population_size = len(parent_population[child_type])
        DNA_size = len(child)
        if np.random.rand() < self.cross_rate:
            i_ = randrange(population_size)  # select another individual from pop
            parent = parent_population[child_type][i_]
            cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)  # choose crossover points
            for (parameter, value), replace in zip(child.items(), cross_points):
                if replace:
                    child[parameter] = parent[parameter] # mating and produce one child
        return child

    def _mutate(self, child: dict):
        """
        Given the mutation probability, randomly mutate a parameter from current child to a parameter
        from all available parameter values
        :param child: Dict containing all parameters for a training run
        :return: mutated child
        """
        child_type = self._get_embedding_key(child)
        for parameter in child.keys():
            if np.random.rand() < self.mutation_rate:
                func = self.all_configurations[child_type][parameter]['method']
                if self.all_configurations[child_type][parameter].get("options") is not None:
                    child[parameter] = func(self.all_configurations[child_type][parameter]['options'])
                elif self.all_configurations[child_type][parameter].get("bounds") is not None:
                    child[parameter] = func(self.all_configurations[child_type][parameter]['bounds'])
        return child

    def _get_embedding_key(self, embedding: dict):
        """
        return depending on document specific parameters the embedding key
        :param embedding:
        :return:
        """
        if self.document_embedding_specific == True:
            embedding_key = embedding['document_embeddings'].__name__
        else:
            embedding_key = "universal_embeddings"
        return embedding_key