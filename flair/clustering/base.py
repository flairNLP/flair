from abc import abstractmethod
from typing import Union, List

from flair.data import Sentence

import torch


class ClusteringModel:

    @abstractmethod
    def fit(self):
        """
        Trains the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        """
        Predict labels given a list of sentences and returns the respective class indices.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Saves current model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """
        Loads a model.
        """
        raise NotImplementedError
