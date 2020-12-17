import logging

from enum import Enum
from abc import abstractmethod

from .budget import Budget
from .parameter_collections import ParameterStorage, TrainingConfigurations
from .parameters import BudgetConstraint, EvaluationMetric, OptimizationValue
from .parameter_groups import DOCUMENT_EMBEDDINGS

log = logging.getLogger("flair")


class SearchSpace(object):
    """
    Search space object, holding all parameters to optimize over as key-value pairs (parent class)
    Implicitly defines desired downstream task when instantiated by child class
    """

    def __init__(self, has_document_embeddings: bool):
        """
        :param has_document_embeddings: bool - True if child class has document embeddings
        """
        self.parameter_storage = ParameterStorage()
        self.training_configurations = TrainingConfigurations()
        self.budget = Budget()
        self.current_run = 0
        self.optimization_value = {}
        self.evaluation_metric = {}
        self.max_epochs_per_training_run = 50
        self.has_document_embeddings = has_document_embeddings

    @abstractmethod
    def add_parameter(self, parameter: Enum, options):
        """
        adds universal parameter to storage object
        :param parameter: Enum parameter object
        :param options: all values the parameter takes
        :return: -
        """
        pass

    @abstractmethod
    def add_word_embeddings(self):
        """
        adds word embeddings to storage obejct
        """
        pass

    @abstractmethod
    def check_completeness(self, search_strategy: str):
        """
        checks if all necessary parameters are set
        """
        pass

    def add_budget(self, budget: BudgetConstraint, amount: int):
        """
        sets budget parameter for optimization procedure
        :param budget: Enum BudgetConstaint object
        :param amount: value for respective budget
        :return: -
        """
        self.budget.add(budget_type=budget.value, amount=amount)

    def add_optimization_value(self, optimization_value: OptimizationValue):
        """
        adds optimization value to storage object
        :param optimization_value: Enum OptimizationValue object
        :return: -
        """
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        """
        adds evaluation metric to storage object
        :param evaluation_metric: Enum EvaluationMetric object
        :return: -
        """
        self.evaluation_metric = evaluation_metric.value

    def add_max_epochs_per_training_run(self, max_epochs: int):
        """
        sets max. epochs for each training run
        :param max_epochs: amount of maximal epochs per training run
        :return: -
        """
        self.max_epochs_per_training_run = max_epochs

    def _check_for_mandatory_steering_parameters(self):
        """
        checks if all mandatory parameters are set, passes if True, otherwise return Exception
        :return: -
        """
        if not all([self.budget, self.optimization_value, self.evaluation_metric]):
            raise AttributeError(
                "Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

        if self.parameter_storage.is_empty():
            raise AttributeError("Parameters haven't been set.")

    def _check_budget_type_matches_search_strategy(self, search_strategy: str):
        """
        checks if budget type matches search strategy, i.e. generations are only valid paired with EvolutionarySearch
        :param search_strategy: str of search strategy
        :return: -
        """
        if 'generations' in self.budget.budget_type and search_strategy != "EvolutionarySearch":
            log.info("Can't assign generations to a an Optimizer which is not a GeneticOptimizer. Switching to runs.")
            self.budget.budget_type = "runs"

    @staticmethod
    def _encode_embeddings_for_serialization(embeddings_list: list) -> list:
        """
        Transforms word embedding object into key value pair like "embedding class":"parameters for class"
        :param embeddings_list: list of embedding object
        :return encoded_embedding_list: list of key value pairs of embeddings in order to save memory
        """
        # Since Word Embeddings take much memory upon creation, we only store the its parameters for further processing
        encoded_embeddings_list = []
        for stacked_embeddings in embeddings_list:
            encoded_stacked_embeddings = [embedding.instance_parameters for embedding in stacked_embeddings]
            encoded_embeddings_list.append(encoded_stacked_embeddings)
        return encoded_embeddings_list


class TextClassifierSearchSpace(SearchSpace):
    """
    Search space for text classification. Is document embedding specific.
    """

    def __init__(self, multi_label: bool = False):
        """
        :param multi_label: bool, True if multi label classification
        :return: -
        """
        super().__init__(has_document_embeddings=True)
        self.multi_label = multi_label

    def add_parameter(self, parameter: Enum, options: list):
        """
        overrides parent function. Adds parameter to storage object
        :param parameter: Enum parameter object
        :param options: all values the parameter takes
        :return: -
        """
        parameter_values = {"value_range": options}
        if self._is_embedding_specific_parameter(parameter):
            parameter_values["embedding_key"] = parameter.__class__.__name__
        self.parameter_storage.add(parameter_name=parameter.value, **parameter_values)

    def add_word_embeddings(self, parameter: Enum, options: list):
        """
        overrides parent function. adds word embeddings to storage object.
        :param parameter: Enum parameter object
        :param options: all values the parameter takes
        :return: -
        """
        encoded_embeddings = self._encode_embeddings_for_serialization(options)
        parameter_values = {"value_range": encoded_embeddings}
        if self._is_embedding_specific_parameter(parameter):
            parameter_values["embedding_key"] = parameter.__class__.__name__
        self.parameter_storage.add(parameter_name="embeddings", **parameter_values)

    @staticmethod
    def _is_embedding_specific_parameter(parameter: Enum) -> bool:
        """
        checks if provided parameter if document embedding specific (due to text classification downstream task)
        :param parameter: Enum Parameter object
        :return: bool - True if DocumentEmbedding specific
        """
        if parameter.__class__.__name__ in DOCUMENT_EMBEDDINGS:
            return True
        else:
            return False

    def check_completeness(self, search_strategy: str):
        """
        override parent function. wrapper function to check if all necessary parameters are set before starting training.
        Passes if everything is set correctly.
        :param search_strategy: str of search strategy
        :return: -
        """
        self._check_for_mandatory_steering_parameters()
        self._check_budget_type_matches_search_strategy(search_strategy)
        self._check_document_embeddings_are_set()

    def _check_document_embeddings_are_set(self):
        """
        checks if document embeddings have been set, necessary for text classification.
        Passes if at least 1 have been set, else raises Exception
        :return: -
        """
        currently_set_parameters = self.parameter_storage.__dict__.keys()

        document_embeddings = [embedding for embedding in currently_set_parameters
                               if embedding in DOCUMENT_EMBEDDINGS]

        if not document_embeddings:
            raise Exception("No document embeddings have been set.")

        for single_embedding in document_embeddings:
            if not bool(getattr(self.parameter_storage, single_embedding).get("embeddings")) \
                    and single_embedding != "TransformerDocumentEmbeddings":
                raise Exception("Please set WordEmbeddings for DocumentEmbeddings.")


class SequenceTaggerSearchSpace(SearchSpace):
    """
    Search space for sequence tagging. Is not document embedding specific.
    """

    def __init__(self):
        super().__init__(has_document_embeddings=False)
        self.tag_type = ""

    def add_tag_type(self, tag_type: str):
        """
        adds tag type for task
        :param tag_type: str of tag type
        :return: -
        """
        self.tag_type = tag_type

    def add_parameter(self, parameter: Enum, options: list):
        """
        overrides parent function. Adds parameter to storage object
        :param parameter: Enum parameter object
        :param options: all values the parameter takes
        :return: -
        """
        self.parameter_storage.add(parameter_name=parameter.value, value_range=options)

    def add_word_embeddings(self, options: list):
        """
        overrides parent function. adds word embeddings to storage object.
        :param parameter: Enum parameter object
        :param options: all values the parameter takes
        :return: -
        """
        encoded_embeddings = self._encode_embeddings_for_serialization(options)
        self.parameter_storage.add(parameter_name="embeddings", value_range=encoded_embeddings)

    def check_completeness(self, search_strategy: str):
        """
        override parent function. wrapper function to check if all necessary parameters are set before starting training.
        Passes if everything is set correctly.
        :param search_strategy: str of search strategy
        :return: -
        """
        self._check_for_mandatory_steering_parameters()
        self._check_budget_type_matches_search_strategy(search_strategy)
        self._check_word_embeddings_are_set()

    def _check_word_embeddings_are_set(self):
        """
        checks if word embeddings have been set, necessary for sequence labeling.
        Passes if at least 1 have been set, else raises Exception
        :return: -
        """
        if self.parameter_storage.general_parameters.get("embeddings"):
            pass
        else:
            raise Exception("Word Embeddings haven't been set.")
