import inspect
import logging
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Sequence, Type, Union

import torch
from torch.nn import Parameter, ParameterList

import flair
from flair.data import DT, Sentence

log = logging.getLogger("flair")


class Embeddings(torch.nn.Module, Generic[DT]):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    embeddings_name: str  # class-variable referring to the "class embedding name"

    def __init__(self) -> None:
        """Set some attributes that would otherwise result in errors. Overwrite these in your embedding class."""
        if not hasattr(self, "name"):
            self.name: str = "unnamed_embedding"
        if not hasattr(self, "static_embeddings"):
            # if the embeddings for a sentence are the same in each epoch, set this to True for improved efficiency
            self.static_embeddings = False
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        raise NotImplementedError

    def embed(self, data_points: Union[DT, List[DT]]) -> List[DT]:
        """Add embeddings to all words in a list of sentences.

        If embeddings are already added, updates only if embeddings are non-static.
        """
        # if only one sentence is passed, convert to list of sentence
        if not isinstance(data_points, list):
            data_points = [data_points]

        if not self._everything_embedded(data_points):
            self._add_embeddings_internal(data_points)

        return data_points

    def _everything_embedded(self, data_points: Sequence[DT]) -> bool:
        return all(self.name in data_point._embeddings for data_point in data_points)

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[DT]):
        """Private method for adding embeddings to all words in a list of sentences."""

    def get_names(self) -> List[str]:
        """Returns a list of embedding names.

        In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack.
        """
        return [self.name]

    def get_named_embeddings_dict(self) -> Dict:
        return {self.name: self}

    @staticmethod
    def get_instance_parameters(locals: dict) -> dict:
        class_definition = locals.get("__class__")
        instance_parameter_names = set(inspect.signature(class_definition.__init__).parameters)  # type: ignore[misc]
        instance_parameter_names.remove("self")
        instance_parameter_names.add("__class__")
        instance_parameters = {
            class_attribute: attribute_value
            for class_attribute, attribute_value in locals.items()
            if class_attribute in instance_parameter_names
        }
        return instance_parameters

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Embeddings":
        raise NotImplementedError

    def to_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def load_embedding(cls, params: Dict[str, Any]):
        state_dict = params.pop("state_dict", None)

        embedding = cls.from_params(params)
        if state_dict is not None:
            embedding.load_state_dict(state_dict)
        return embedding

    def save_embeddings(self, use_state_dict: bool = True):
        params = self.to_params()
        if use_state_dict:
            params["state_dict"] = self.state_dict()
        params["__cls__"] = type(self).embeddings_name
        return params


class ScalarMix(torch.nn.Module):
    """Mixes several tensors by a learned weighting.

    Computes a parameterised scalar mixture of N tensors.
    This method was proposed by Liu et al. (2019) in the paper:
    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)

    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.
    It can be found under:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.
    """

    def __init__(self, mixture_size: int, trainable: bool = False) -> None:
        """Inits scalar mix implementation.

        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

        Args:
            mixture_size: size of mixtures (usually the number of layers)
            trainable: weather or not the weights should be learnable.
        """
        super().__init__()
        self.mixture_size = mixture_size

        initial_scalar_parameters = [0.0] * mixture_size

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.tensor(
                        [initial_scalar_parameters[i]],
                        dtype=torch.float,
                        device=flair.device,
                    ),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(
            torch.tensor(
                [1.0],
                dtype=torch.float,
                device=flair.device,
            ),
            requires_grad=trainable,
        )

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of scalar mix.

        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        Args:
            tensors: list of input tensors

        Returns: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error(
                f"{len(tensors)} tensors were passed, but the module was initialized to mix {self.mixture_size} tensors."
            )

        normed_weights = torch.nn.functional.softmax(torch.cat(list(self.scalar_parameters)), dim=0)
        normed_weights_split = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights_split, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class DocumentEmbeddings(Embeddings[Sentence]):
    """Abstract base class for all document-level embeddings. Every new type of document embedding must implement these methods."""

    @property
    def embedding_type(self) -> str:
        return "sentence-level"


class TokenEmbeddings(Embeddings[Sentence]):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    def embedding_type(self) -> str:
        return "word-level"

    def _everything_embedded(self, data_points: Sequence[Sentence]) -> bool:
        for sentence in data_points:
            for token in sentence.tokens:
                if self.name not in token._embeddings:
                    return False
        return True


EMBEDDING_CLASSES: Dict[str, Type[Embeddings]] = {}


def register_embeddings(*args):
    name = None

    def _register(cls):
        nonlocal name
        if name is None:
            name = cls.__name__
        cls.embeddings_name = name
        EMBEDDING_CLASSES[name] = cls
        return cls

    if len(args) == 1 and callable(args[0]):
        return _register(args[0])
    elif len(args) > 0:
        name = args[0]

    return _register


def load_embeddings(params: Dict[str, Any]) -> Embeddings:
    cls_name = params.pop("__cls__")
    cls = EMBEDDING_CLASSES[cls_name]
    return cls.load_embedding(params)
