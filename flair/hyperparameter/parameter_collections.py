import itertools
import random
from .parameter_groups import DOCUMENT_EMBEDDINGS


class ParameterStorage():

    def __init__(self):
        pass

    def add(self, parameter_name: str, value_range: list, embedding_key : str  = "GeneralParameters"):
        if hasattr(self, embedding_key):
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)
        else:
            self._create_new_embedding_key(embedding_key)
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)

    def _create_new_embedding_key(self, parameter_name: str):
        if parameter_name in DOCUMENT_EMBEDDINGS:
            setattr(self, parameter_name, {"document_embeddings":[eval(parameter_name)]})
        else:
            setattr(self, parameter_name, {})

    def _append_to_existing_embedding_key(self, embedding_key: str, parameter_name: str, parameter: dict):
        getattr(self, embedding_key)[parameter_name] = parameter

    def is_empty(self):
        if not bool(self.__dict__):
            return True


class TrainingConfigurations():

    def __init__(self):
        self.configurations = []

    def get_configuration(self):
        return self.configurations.pop(0)

    def make_grid_configurations(self, parameter_storage: ParameterStorage):
        parameters_per_embedding_list = self._get_parameters_per_embedding(parameter_storage)
        grid_configurations = self._make_cartesian_product(parameters_per_embedding_list)
        self.configurations.extend(grid_configurations)

    def _get_parameters_per_embedding(self, parameter_storage: ParameterStorage):
        embedding_specific_keys_in_parameter_storage, general_parameters = self._get_parameter_keys(parameter_storage)
        if embedding_specific_keys_in_parameter_storage:
            parameter_dictionaries = self._make_embedding_specific_parameter_dictionaries(embedding_specific_keys_in_parameter_storage,
                                                                          general_parameters,
                                                                          parameter_storage)
        else:
            parameter_dictionaries = self._make_parameter_dictionary(general_parameters, parameter_storage)
        return parameter_dictionaries

    def _get_parameter_keys(self, parameter_storage: ParameterStorage):
        embedding_specific_keys_in_parameter_storage = parameter_storage.__dict__.keys() & DOCUMENT_EMBEDDINGS
        general_parameter_keys = parameter_storage.__dict__.keys() - DOCUMENT_EMBEDDINGS
        return embedding_specific_keys_in_parameter_storage, general_parameter_keys

    def _make_embedding_specific_parameter_dictionaries(self, embedding_keys: set, general_keys: set, parameter_storage: ParameterStorage) -> list:
        list_of_parameter_dictionaries = []
        for embedding_key, general_key in itertools.product(embedding_keys, general_keys):
            embedding_specific_parameters = getattr(parameter_storage, embedding_key)
            general_parameters = getattr(parameter_storage, general_key)
            complete_parameters_per_embedding = {**embedding_specific_parameters, **general_parameters}
            list_of_parameter_dictionaries.append(complete_parameters_per_embedding)
        return list_of_parameter_dictionaries

    def _make_parameter_dictionary(self, general_keys: set, parameter_storage: ParameterStorage) -> list:
        list_of_parameter_dictionaries = []
        general_key = general_keys.pop()
        general_parameters = getattr(parameter_storage, general_key)
        list_of_parameter_dictionaries.append(general_parameters)
        return list_of_parameter_dictionaries

    def _make_cartesian_product(self, parametersList: list):
        cartesian_product = []
        for parameters in parametersList:
            keys, values = zip(*parameters.items())
            training_configurations = itertools.product(*values)
            for configuration in training_configurations:
                cartesian_product.append(dict(zip(keys, configuration)))
        return cartesian_product

    def make_evolutionary_configurations(self, parameter_storage: ParameterStorage, number_of_configurations: int):
        parameters_per_embedding_list = self._get_parameters_per_embedding(parameter_storage)
        number_of_different_embedding_types = len(parameters_per_embedding_list)
        configurations_per_embedding_type = self._equally_distribute_configurations_per_embedding(number_of_different_embedding_types,
                                                                                                  number_of_configurations)
        for parameter_dictionary, amount_configurations in zip(parameters_per_embedding_list, configurations_per_embedding_type):
            # typecasting needed here since cartesian_product only accepts list of parameter dictionaries
            # but we want to sample equally from each embedding and append it to our initial population
            entire_configuration = self._make_cartesian_product([parameter_dictionary])
            evolutionary_configurations = random.sample(entire_configuration, amount_configurations)
            self.configurations.extend(evolutionary_configurations)

    def _equally_distribute_configurations_per_embedding(self, number_of_different_embeddings: int, number_of_configurations: int) -> list:
        configurations_per_embedding = [number_of_configurations // number_of_different_embeddings +
                                        (1 if x < number_of_configurations % number_of_different_embeddings else 0)
                                        for x in range(number_of_different_embeddings)]
        return configurations_per_embedding

    def _add_configuration(self, configuration: dict):
        self.configurations.append(configuration)