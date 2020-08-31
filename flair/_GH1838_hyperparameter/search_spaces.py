import time
from enum import Enum
from abc import abstractmethod

from .sampling_functions import func
from .parameters import Budget, EvaluationMetric, OptimizationValue

"""
The Search Space object acts as a data object containing all configurations for the hyperparameter optimization.
We currently support two types of downstream task for hyperparameter optimization:
    Text Classification
    Sequence Labeling
    
Steering parameters which have to bet set independent of downstream task:
    Steering params:                        function to use:
    A budget preventing a long runtime      add_budget()
    A evaluation metric for training        add_evaluation_metric()
    An optimization value for training      add_optimization_value()
    Max epochs per training run             add_max_epochs_training() (default: 50)
    
For text classification, please first set document embeddings since you probably add document specific embeddings

Add parameters like this:

import GeneticParamOptimizer.hyperparameter.parameters as param
from GeneticParamOptimizer.hyperparameter.utils import func

search_space.add_parameter(param.[TYPE OF PARAMETER TO BE SET].[CONCRETE PARAMETER],
                           func.[FUNCTION TO PICK FROM VALUE RANGE],
                           options=[LIST OF PARAMETER VALUES] or range=[BOUNDS OF PARAMETER VALUES])

Note following combinations of functions and type of parameter values are possible:

    function:   value range argument:   explanation:
    choice      options=[1,2,3]         choose from different options
    uniform     bounds=[0, 0.5]         take a uniform sample between lower and upper bound
"""

class SearchSpace(object):
    """
    Search space main class.

    Attributes:
        parameters                  Parameters of all configurations
        budget                      Budget for the hyperparameter optimization
        optimization_value          Metric which will be optimized during training
        evaluation_metric           Metric which is used for selecting the best configuration
        max_epochs_per_training     max. number of iterations per training for a single configuration
    """

    def __init__(self, document_embedding_specific: bool):
        self.parameters = {}
        self.budget = {}
        self.optimization_value = {}
        self.evaluation_metric = {}
        self.max_epochs_per_training = 50
        self.document_embedding_specific = document_embedding_specific

    @abstractmethod
    def add_parameter(self,
                      parameter: Enum,
                      func: func,
                      **kwargs):
        """
        Adds single parameter configuration to search space. Overwritten by child class.
        :param parameter: passed
        :param func: passed
        :param kwargs: passed
        :return: passed
        """
        pass

    def add_budget(self,
                   budget: Budget,
                   value):
        """
        Adds a budget for the entire hyperparameter optimization.
        :param budget: Type of budget which is going to be used
        :param value: Budget value - depending on budget type
        :return: none
        """
        self.budget[budget.value] = value

        if budget.value == "time_in_h":
            self.start_time = time.time()

    def add_optimization_value(self, optimization_value: OptimizationValue):
        """
        Adds optimization value to the search space.
        :param optimization_value: Optimization Value from Enum class.
        :return: none
        """
        self.optimization_value = optimization_value.value

    def add_evaluation_metric(self, evaluation_metric: EvaluationMetric):
        """
        Sets evaluation metric for training
        :param evaluation_metric:
        :return:
        """
        self.evaluation_metric = evaluation_metric.value

    def add_max_epochs_per_training(self, max_epochs: int):
        """
        Set max iteration per training for a single configuration
        :param max_epochs:
        :return:
        """
        self.max_epochs_per_training = max_epochs

    def _check_function_param_match(self, kwargs):
        """
        Checks whether options or bounds are provided as value search space.
        :param kwargs:
        :return:
        """
        if len(kwargs) != 1 and \
                not "options" in kwargs and \
                not "bounds" in kwargs:
            raise Exception("Please provide either options or bounds depending on your function.")

    def _check_document_embeddings_are_set(self, parameter: Enum):
        """
        Checks whether Document Embeddings have been. They need to come first in order to assign Document specific parameters.
        :param parameter: Parameter to be set
        :return: None
        """
        if self.document_embedding_specific == False:
            print("Warning: You can only check for search spaces which have document embeddings specific parameters.")
            return

        if not self.parameters and parameter.name != "DOCUMENT_EMBEDDINGS":
            raise Exception("Please provide first the document embeddings in order to assign model specific attributes")

    def _check_mandatory_parameters_are_set(self, optimizer_type: str):
        if not all([self.budget, self.parameters, self.optimization_value, self.evaluation_metric]) \
                and self._check_budget_type(optimizer_type):
            raise Exception("Please provide a budget, parameters, a optimization value and a evaluation metric for an optimizer.")

    def _check_budget_type(self, optimizer_type):
        if 'generations' in self.budget and optimizer_type == "GeneticOptimizer":
            return True
        elif 'runs' in self.budget or 'time_in_h' in self.budget:
            return True
        else:
            return False

class TextClassifierSearchSpace(SearchSpace):
    """
    Search space for the text classification downstream task

    Attributes:
        inherited from SearchSpace object
    """

    def __init__(self):
        super().__init__(
            document_embedding_specific=True
        )

    def add_parameter(self,
                      parameter: Enum,
                      func: func,
                      **kwargs):
        """
        Adds configuration for a single parameter to the search space.
        :param parameter: Type of parameter
        :param func: Function how to choose values from the parameter configuration
        :param kwargs: Either options or bounds depending on the function
        :return: None
        """
        self._check_function_param_match(kwargs)

        # This needs to be checked here,
        # since we want to set document embeddings specific parameters
        self._check_document_embeddings_are_set(parameter)

        if parameter.name == "DOCUMENT_EMBEDDINGS":
            self._add_document_embeddings(parameter, func, **kwargs)
        else:
            self._add_parameters(parameter, func, kwargs)


    def _add_document_embeddings(self,
                                 parameter: Enum,
                                 func: func,
                                 options):
        """
        Adds document embeddings to search space.
        :param parameter: Document Embedding to be set
        :param func: Function to pick from value range
        :param options: Value range
        :return: None
        """
        try:
            for embedding in options:
                self.parameters[embedding.__name__] = {parameter.value: {"options": [embedding], "method": func}}
        except:
            raise Exception("Document embeddings only takes options as arguments")


    def _add_parameters(self,
                        parameter: Enum,
                        func: func,
                        kwargs):
        """
        Wrapper function to add document embedding specific parameter or universal parameter
        :param parameter: Parameter to be set
        :param func: Function to pick from value range
        :param kwargs: Value range
        :return: None
        """
        if "Document" in parameter.__class__.__name__:
            self._add_embedding_specific_parameter(parameter, func, kwargs)
        else:
            self._add_universal_parameter(parameter, func, kwargs)

    def _add_embedding_specific_parameter(self,
                                          parameter: Enum,
                                          func: func,
                                          kwargs):
        """
        Adds a document embedding specific parameter.
        :param parameter: Parameter to be set
        :param func: Function to pick from value range
        :param kwargs: Value range
        :return: None
        """
        try:
            for key, values in kwargs.items():
                self.parameters[parameter.__class__.__name__].update({parameter.value: {key: values, "method": func}})
        except:
            raise Exception("If your want to assign document embedding specific parameters, make sure it is included in the search space.")

    def _add_universal_parameter(self,
                                 parameter: Enum,
                                 func: func,
                                 kwargs):
        """
        Adds a universal training parameter independent from the document embeddings
        :param parameter: Parameter to be set
        :param func: Function to pick from value range
        :param kwargs: Value range
        :return: None
        """
        for embedding in self.parameters:
            for key, values in kwargs.items():
                self.parameters[embedding].update({parameter.value: {key: values, "method": func}})


class SequenceTaggerSearchSpace(SearchSpace):
    """
    Search space for the sequence tagging downstream task

    Attributes:
        tag_type    Type of sequence labels
    """

    def __init__(self):
        super().__init__(
            document_embedding_specific=False
        )

        self.tag_type = ""

    def add_tag_type(self, tag_type : str):
        """
        Adds the tag type to the search space object
        :param tag_type: Tag type from corpus
        :return: None
        """
        self.tag_type = tag_type

    def add_parameter(self,
                      parameter: Enum,
                      func: func,
                      **kwargs):
        """
        Adds parameter to the
        :param parameter:
        :param func:
        :param kwargs:
        :return:
        """
        for key, values in kwargs.items():
            self.parameters.update({parameter.value : {key: values, "method": func}})