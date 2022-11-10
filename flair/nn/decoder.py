import logging
from typing import Optional

import torch

import flair
from flair.nn.distance import (
    CosineDistance,
    EuclideanDistance,
    HyperbolicDistance,
    LogitCosineDistance,
    NegativeScaledDotProduct,
)

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
    ):

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
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded

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
