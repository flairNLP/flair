import torch.nn

from abc import abstractmethod

from typing import Union, List

from flair.data import Sentence, Label


class Model(torch.nn.Module):
    """Abstract base class for all models. Every new type of model must implement these methods."""

    @abstractmethod
    def forward_and_loss(self, sentences: Union[List[Sentence], Sentence]) -> float:
        """Performs a forward pass and returns the loss."""
        pass

    @abstractmethod
    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:
        """Predicts the labels/tags for the given list of sentences. The labels/tags are added directly to the
        sentences."""
        pass

    @abstractmethod
    def predict_eval(self, sentences: Union[List[Sentence], Sentence]) -> (List[List[Label]], float):
        """Predicts the labels/tags for the given list of sentences. Returns the list of labels plus the loss."""
        pass


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """
    def __init__(self, dropout_rate=0.05):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), 1, 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x