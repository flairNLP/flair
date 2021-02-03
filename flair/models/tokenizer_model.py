import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data.dataset import Dataset

import flair
from flair.data import DataPoint
from flair.training_utils import Result


class FlairTokenizer(flair.nn.Model):

    def __init__(self,
                 character_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 tagset_size,
                 batch_size,
                 use_CSE=False,
                 ):

        super(FlairTokenizer, self).__init__()

        # TODO: This init method should be the same as your LSTMTagger
        pass

    @abstractmethod
    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""

        # TODO: what is currently your forward() goes here, followed by the loss computation
        # Since the DataPoint brings its own label, you can compute the loss here
        pass

    @abstractmethod
    def evaluate(
        self,
        sentences: Union[List[DataPoint], Dataset],
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """

        # TODO: Your evaluation routine goes here. For the DataPoints passed into this method, compute the accuracy
        # and store it in a Result object, which you return.

        pass

    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""
        pass

    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        pass

    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name