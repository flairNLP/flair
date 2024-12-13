import logging
from typing import Optional

import torch

import flair
from flair.data import Dictionary, Sentence
from flair.embeddings import Embeddings
from flair.nn.distance import (
    CosineDistance,
    EuclideanDistance,
    HyperbolicDistance,
    LogitCosineDistance,
    NegativeScaledDotProduct,
)
from flair.training_utils import store_embeddings

logger = logging.getLogger("flair")


class PrototypicalDecoder(torch.nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        embeddings_size: int,
        prototype_size: Optional[int] = None,
        distance_function: str = "euclidean",
        use_radius: Optional[bool] = False,
        min_radius: Optional[int] = 0,
        unlabeled_distance: Optional[float] = None,
        unlabeled_idx: Optional[int] = None,
        learning_mode: Optional[str] = "joint",
        normal_distributed_initial_prototypes: bool = False,
    ) -> None:
        super().__init__()

        if not prototype_size:
            prototype_size = embeddings_size

        self.prototype_size = prototype_size

        # optional metric space decoder if prototypes have different length than embedding
        self.metric_space_decoder: Optional[torch.nn.Linear] = None
        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size, prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)

        # create initial prototypes for all classes (all initial prototypes are a vector of all 1s)
        self.prototype_vectors = torch.nn.Parameter(torch.ones(num_prototypes, prototype_size), requires_grad=True)

        # if set, create initial prototypes from normal distribution
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(num_prototypes, prototype_size)))

        # if set, use a radius
        self.prototype_radii: Optional[torch.nn.Parameter] = None
        if use_radius:
            self.prototype_radii = torch.nn.Parameter(torch.ones(num_prototypes), requires_grad=True)

        self.min_radius = min_radius
        self.learning_mode = learning_mode

        assert (unlabeled_idx is None) == (
            unlabeled_distance is None
        ), "'unlabeled_idx' and 'unlabeled_distance' should either both be set or both not be set."

        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_distance = unlabeled_distance

        self._distance_function = distance_function

        self.distance: Optional[torch.nn.Module] = None
        if distance_function.lower() == "hyperbolic":
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == "cosine":
            self.distance = CosineDistance()
        elif distance_function.lower() == "logit_cosine":
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == "euclidean":
            self.distance = EuclideanDistance()
        elif distance_function.lower() == "dot_product":
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f"Distance function {distance_function} not found.")

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded):
        if self.learning_mode == "learn_only_map_and_prototypes":
            embedded = embedded.detach()

        # decode embeddings into prototype space
        encoded = self.metric_space_decoder(embedded) if self.metric_space_decoder is not None else embedded

        prot = self.prototype_vectors
        radii = self.prototype_radii

        if self.learning_mode == "learn_only_prototypes":
            encoded = encoded.detach()

        if self.learning_mode == "learn_only_embeddings_and_map":
            prot = prot.detach()

            if radii is not None:
                radii = radii.detach()

        distance = self.distance(encoded, prot)

        if radii is not None:
            distance /= self.min_radius + torch.nn.functional.softplus(radii)

        # if unlabeled distance is set, mask out loss to unlabeled class prototype
        if self.unlabeled_distance:
            distance[..., self.unlabeled_idx] = self.unlabeled_distance

        scores = -distance

        return scores


class LabelVerbalizerDecoder(torch.nn.Module):
    """A class for decoding labels using the idea of siamese networks / bi-encoders. This can be used for all classification tasks in flair.

    Args:
        label_encoder (flair.embeddings.TokenEmbeddings):
            The label encoder used to encode the labels into an embedding.
        label_dictionary (flair.data.Dictionary):
            The label dictionary containing the mapping between labels and indices.

    Attributes:
        label_encoder (flair.embeddings.TokenEmbeddings):
            The label encoder used to encode the labels into an embedding.
        label_dictionary (flair.data.Dictionary):
            The label dictionary containing the mapping between labels and indices.

    Methods:
        forward(self, label_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
            Takes the label embeddings and context embeddings as input and returns a tensor of label scores.

    Examples:
        label_dictionary = corpus.make_label_dictionary("ner")
        label_encoder = TransformerWordEmbeddings('bert-base-ucnased')
        label_verbalizer_decoder = LabelVerbalizerDecoder(label_encoder, label_dictionary)
    """

    def __init__(self, label_embedding: Embeddings, label_dictionary: Dictionary):
        super().__init__()
        self.label_embedding = label_embedding
        self.verbalized_labels: list[Sentence] = self.verbalize_labels(label_dictionary)
        self.to(flair.device)

    @staticmethod
    def verbalize_labels(label_dictionary: Dictionary) -> list[Sentence]:
        """Takes a label dictionary and returns a list of sentences with verbalized labels.

        Args:
            label_dictionary (flair.data.Dictionary): The label dictionary to verbalize.

        Returns:
            A list of sentences with verbalized labels.

        Examples:
            label_dictionary = corpus.make_label_dictionary("ner")
            verbalized_labels = LabelVerbalizerDecoder.verbalize_labels(label_dictionary)
            print(verbalized_labels)
            [Sentence: "begin person", Sentence: "inside person", Sentence: "end person", Sentence: "single org", ...]
        """
        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            str_label = byte_label.decode("utf-8")
            if label_dictionary.span_labels:
                # verbalize BIOES labels
                if str_label == "O":
                    verbalized_labels.append("outside")
                elif str_label.startswith("B-"):
                    verbalized_labels.append("begin " + str_label.split("-")[1])
                elif str_label.startswith("I-"):
                    verbalized_labels.append("inside " + str_label.split("-")[1])
                elif str_label.startswith("E-"):
                    verbalized_labels.append("ending " + str_label.split("-")[1])
                elif str_label.startswith("S-"):
                    verbalized_labels.append("single " + str_label.split("-")[1])
                # if label is not BIOES, use label itself
                else:
                    verbalized_labels.append(str_label)
            else:
                verbalized_labels.append(str_label)
        return list(map(Sentence, verbalized_labels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the label verbalizer decoder.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            The scores of the decoder.

        Raises:
            RuntimeError: If an unknown decoding type is specified.
        """
        if self.training or not self.label_embedding._everything_embedded(self.verbalized_labels):
            self.label_embedding.embed(self.verbalized_labels)

        label_tensor = torch.stack([label.get_embedding() for label in self.verbalized_labels])

        if self.training:
            store_embeddings(self.verbalized_labels, "none")

        scores = torch.mm(inputs, label_tensor.T)

        return scores
