import itertools
import random

from flair.embeddings.document import *

from .parameter_groups import DOCUMENT_EMBEDDINGS


class ParameterStorage:
    """
    Storage object containing all parameters to optimize over
    """

    def __init__(self):
        pass

    def add(self, parameter_name: str, value_range: list, embedding_key: str = "GeneralParameters"):
        """
        adds a parameter to the storage object
        :param parameter_name: str of parameter
        :param value_range: list of all values the parameter can take during optimization
        :param embedding_key: default is GeneralParameters. There are embedding specific keys, i.e. for DocumentRNNEmbeddings
        """
        if hasattr(self, embedding_key):
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)
        else:
            self._create_new_embedding_key(embedding_key)
            self._append_to_existing_embedding_key(embedding_key, parameter_name, value_range)

    def _create_new_embedding_key(self, embedding_key: str):
        """
        Creates embedding keys in ParameterStorage Object
        :param embedding_key: str of embedding key to be set
        :return: -
        """
        if embedding_key in DOCUMENT_EMBEDDINGS:
            setattr(self, embedding_key, {"document_embeddings": [eval(embedding_key)]})
        else:
            setattr(self, embedding_key, {})

    def _append_to_existing_embedding_key(self, embedding_key: str, parameter_name: str, parameter: dict):
        """
        Appends parameter configuration to existing embedding key
        :return: -
        """
        getattr(self, embedding_key)[parameter_name] = parameter

    def is_empty(self):
        if not bool(self.__dict__):
            return True


class TrainingConfigurations:
    """
    Object holding all training configurations. A single training configuration is passed to the train method.
    In contradiction to parameter storage, where all possible values of a parameter are attached to its respective parameter key.
    """

    def __init__(self):
        self.configurations = []

    def has_configurations_left(self) -> bool:
        """
        Checks if configurations are left. Necessary if we only have a tiny amount of configuration and budget is not used up.
        :return: True if configurations are left.
        """
        if self.configurations:
            return True
        else:
            return False

    def get_configuration(self):
        """
        Pops a configuration from list.
        :return: training run configuration
        """
        return self.configurations.pop(0)

    def make_grid_configurations(self, parameter_storage: ParameterStorage, has_document_embeddings: bool):
        """
        combines all parameters from storage object with each other in order to obtain all possible configurations ot optimize over
        :param parameter_storage: ParameterStorage object
        :param has_document_embeddings: bool determining if storage object has document embeddings in it
        :return: -
        """
        embedding_specific_keys, general_keys = self._split_storage_keys(parameter_storage)
        if has_document_embeddings:
            formatted_parameters = self._combine_general_and_embedding_parameters(embedding_specific_keys, general_keys,
                                                                                  parameter_storage)
        else:
            formatted_parameters = self._format_only_general_parameters(general_keys, parameter_storage)
        grid_configurations = self._make_cartesian_product(formatted_parameters)
        self.configurations.extend(grid_configurations)

    @staticmethod
    def _make_cartesian_product(parameters: list):
        """
        applies cartesian product to parameter list in order to obtain all possible configurations
        :param parameters: list containing the parameters
        :return: cartesian product of input list
        """
        cartesian_product = []
        for parameter_dict in parameters:
            keys, values = zip(*parameter_dict.items())
            training_configurations = itertools.product(*values)
            for configuration in training_configurations:
                cartesian_product.append(dict(zip(keys, configuration)))
        return cartesian_product

    def make_evolutionary_configurations(self, parameter_storage: ParameterStorage, has_document_embeddings: bool,
                                         number_of_configurations: int):
        """
        makes initial evolutionary configurations from parameter storage. same as grid configurations but samples configurations
        of size of population. ensures that initially are embedding specific configurations are present.
        :param parameter_storage: ParameterStorage object
        :param has_document_embeddings: bool defining if storage has embedding specific parameters
        :param number_of_configurations: number of configurations for one generation
        :return: -
        """
        embedding_specific_keys, general_keys = self._split_storage_keys(parameter_storage)

        if has_document_embeddings:
            formatted_parameters = self._combine_general_and_embedding_parameters(embedding_specific_keys, general_keys,
                                                                                  parameter_storage)
        else:
            formatted_parameters = self._format_only_general_parameters(general_keys, parameter_storage)

        number_of_different_embedding_types = len(formatted_parameters)
        configurations_per_embedding_type = self._equally_distribute_configurations_per_embedding(
            number_of_different_embedding_types,
            number_of_configurations)

        for parameter_dictionary, amount_configurations in zip(formatted_parameters, configurations_per_embedding_type):
            # typecasting needed here since cartesian_product only accepts list of parameter dictionaries
            # but we want to sample equally from each embedding and append it to our initial population
            entire_configuration = self._make_cartesian_product([parameter_dictionary])
            evolutionary_configurations = random.sample(entire_configuration, amount_configurations)
            self.configurations.extend(evolutionary_configurations)

    @staticmethod
    def _split_storage_keys(parameter_storage: ParameterStorage):
        """
        splits parameters from parameter storage whether they are document embedding specific or not. Necessary preprocess for making training configurations.
        :param parameter_storage: ParameterStorage object
        :return: embedding specific keys, general keys
        """
        embedding_specific_keys = parameter_storage.__dict__.keys() & DOCUMENT_EMBEDDINGS
        general_keys = parameter_storage.__dict__.keys() - DOCUMENT_EMBEDDINGS
        return embedding_specific_keys, general_keys

    @staticmethod
    def _combine_general_and_embedding_parameters(embedding_specific_keys: set, general_keys: set,
                                                  parameter_storage: ParameterStorage):
        """
        combines general and embedding specific parameters since all general keys have to be used in a training run,
        but only with one embedding specific configuration.
        for instance: there are two embedding specific parameters, i.e. for DocumentRNNEmbeddings and for TransformerWordEmbeddings.
        We want to ensure general parameters as learning rate is used for both of them, but we want to avoid that TransformerWordEmbedding
        parameters are used during a training run of DocumentRNNEmbeddings.
        :param embedding_specific_keys: embedding specific keys
        :param general_keys: general keys
        :param parameter_storage: ParameterStorage object
        :return: (General Keys + Embedding Specifc Keys (1)), ..., (General Keys + Embedding Specific Keys (n)) where n is the number of embedding specific keys
        """
        combined_parameters = []
        for embedding_key, general_key in itertools.product(embedding_specific_keys, general_keys):
            embedding_specific_parameters = getattr(parameter_storage, embedding_key)
            general_parameters = getattr(parameter_storage, general_key)
            combined_parameters_single_embedding = {**embedding_specific_parameters, **general_parameters}
            combined_parameters.append(combined_parameters_single_embedding)
        return combined_parameters

    @staticmethod
    def _format_only_general_parameters(general_keys: set, parameter_storage: ParameterStorage) -> list:
        """
        Applies formatting to general parameters to use them as training configurations.
        :param general_keys: set of general keys
        :param parameter_storage: ParameterStorage object
        :return: formatted parameters as list
        """
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

    @staticmethod
    def _get_number_of_different_embeddings(combined_parameters: list, has_document_embeddings: bool):
        """
        extracts number of different embeddings in parameter list
        :param combined_parameters: list containing all parameters
        :param has_document_embeddings: bool if list has document embeddings specific parameters
        :return: amount of embedding specific parameters
        """

        # If the length of combined parameters is only 1, we need to check if there are only word embeddings set
        # in the search space in order to ensure that we take each word embedding in the initial population
        if has_document_embeddings:
            return len(combined_parameters)
        else:
            parameters = combined_parameters[0]
            if bool(parameters.get("embeddings")):
                return len(parameters.get("embeddings"))
            else:
                return len(combined_parameters)

    @staticmethod
    def _equally_distribute_configurations_per_embedding(number_of_different_embeddings: int,
                                                         number_of_configurations: int) -> list:
        """
        Returns a equally distributed list, based on how many configuration per generation there are and how many embedding specific keys
        :param number_of_different_embeddings: number of different embeddings
        :param number_of_configurations: population size of one generation
        :return: list of ints - i.e. split 10 into 3 chunks returns list in format [4,3,3]
        """
        configurations_per_embedding = [number_of_configurations // number_of_different_embeddings +
                                        (1 if x < number_of_configurations % number_of_different_embeddings else 0)
                                        for x in range(number_of_different_embeddings)]
        return configurations_per_embedding

    def add_configuration(self, configuration: dict):
        self.configurations.append(configuration)
