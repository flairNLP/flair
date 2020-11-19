import itertools
import random

from flair.embeddings.document import *

from .parameter_groups import DOCUMENT_EMBEDDINGS


class ParameterStorage:

    def __init__(self):
        self.GeneralParameters = {}

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


class TrainingConfigurations:

    def __init__(self):
        self.configurations = []

    def get_configuration(self):
        return self.configurations.pop(0)

    def make_grid_configurations(self, parameter_storage: ParameterStorage, has_document_embeddings: bool):
        embedding_specific_keys, general_keys = self._split_storage_keys(parameter_storage)
        if has_document_embeddings:
            formatted_parameters = self._combine_general_and_embedding_parameters(embedding_specific_keys, general_keys, parameter_storage)
        else:
            formatted_parameters = self._format_only_general_parameters(general_keys, parameter_storage)
        grid_configurations = self._make_cartesian_product(formatted_parameters)
        self.configurations.extend(grid_configurations)

    def _make_cartesian_product(self, parameters: list):
        cartesian_product = []
        for parameter_dict in parameters:
            keys, values = zip(*parameter_dict.items())
            training_configurations = itertools.product(*values)
            for configuration in training_configurations:
                cartesian_product.append(dict(zip(keys, configuration)))
        return cartesian_product

    def make_evolutionary_configurations(self, parameter_storage: ParameterStorage, has_document_embeddings: bool, number_of_configurations: int):
        embedding_specific_keys, general_keys = self._split_storage_keys(parameter_storage)

        if has_document_embeddings:
            formatted_parameters = self._combine_general_and_embedding_parameters(embedding_specific_keys, general_keys, parameter_storage)
        else:
            formatted_parameters = self._format_only_general_parameters(general_keys, parameter_storage)

        number_of_different_embedding_types = len(formatted_parameters)
        configurations_per_embedding_type = self._equally_distribute_configurations_per_embedding(number_of_different_embedding_types,
                                                                                                  number_of_configurations)

        for parameter_dictionary, amount_configurations in zip(formatted_parameters, configurations_per_embedding_type):
            # typecasting needed here since cartesian_product only accepts list of parameter dictionaries
            # but we want to sample equally from each embedding and append it to our initial population
            entire_configuration = self._make_cartesian_product([parameter_dictionary])
            evolutionary_configurations = random.sample(entire_configuration, amount_configurations)
            self.configurations.extend(evolutionary_configurations)

    def _split_storage_keys(self, parameter_storage: ParameterStorage):
        embedding_specific_keys = parameter_storage.__dict__.keys() & DOCUMENT_EMBEDDINGS
        general_keys = parameter_storage.__dict__.keys() - DOCUMENT_EMBEDDINGS
        return embedding_specific_keys, general_keys

    def _combine_general_and_embedding_parameters(self, embedding_specific_keys: set, general_keys: set, parameter_storage: ParameterStorage):
        combined_parameters = []
        for embedding_key, general_key in itertools.product(embedding_specific_keys, general_keys):
            embedding_specific_parameters = getattr(parameter_storage, embedding_key)
            general_parameters = getattr(parameter_storage, general_key)
            combined_parameters_single_embedding = {**embedding_specific_parameters, **general_parameters}
            combined_parameters.append(combined_parameters_single_embedding)
        return combined_parameters

    def _format_only_general_parameters(self, general_keys: set, parameter_storage: ParameterStorage) -> list:
        formatted_parameters = []
        general_key = general_keys.pop()
        general_parameters = getattr(parameter_storage, general_key)
        if bool(general_parameters.get("embeddings")):
            embeddings = general_parameters.pop("embeddings")
            for each_embedding in embeddings:
                parameters = general_parameters.copy()
                parameters["embeddings"] = [each_embedding]
                formatted_parameters.append(parameters)
        else:
            formatted_parameters.append(general_parameters)
        return formatted_parameters

    def _get_number_of_different_embeddings(self, combined_parameters: list, has_document_embeddings: bool):
        # If the length of combined parameters is only 1, we need to check if there are only word embeddings set in the search space
        # in order to ensure that we take each word embedding in the initial population
        if has_document_embeddings:
            return len(combined_parameters)
        else:
            parameters = combined_parameters[0]
            if bool(parameters.get("embeddings")):
                return len(parameters.get("embeddings"))
            else:
                return len(combined_parameters)

    def _equally_distribute_configurations_per_embedding(self, number_of_different_embeddings: int, number_of_configurations: int) -> list:
        configurations_per_embedding = [number_of_configurations // number_of_different_embeddings +
                                        (1 if x < number_of_configurations % number_of_different_embeddings else 0)
                                        for x in range(number_of_different_embeddings)]
        return configurations_per_embedding

    def _add_configuration(self, configuration: dict):
        self.configurations.append(configuration)