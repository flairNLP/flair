import torch

import flair
from flair.data import FlairDataset
from flair.nn.model import DefaultClassifier
from flair.datasets import DataLoader

from .distance import (HyperbolicDistance, CosineDistance, LogitCosineDistance,
                       EuclideanDistance, NegativeScaledDotProduct)

from tqdm import tqdm

from collections import Counter

from typing import Optional, List

import logging

logger = logging.getLogger('flair')


class PrototypicalDecoder(torch.nn.Module):
    def __init__(self,
        num_prototypes: int,
        embeddings_size : int,
        prototype_size : Optional[int] = None,
        distance_function: Optional[str] = 'euclidean',
        use_radius : Optional[bool] = False,
        min_radius : Optional[int] = 0,
        unlabeled_distance: Optional[float] = None,
        unlabeled_idx : Optional[int] = None,
        learning_mode : Optional[str] = 'joint',
        normal_distributed_initial_prototypes: bool = False,):

        super().__init__()

        if not prototype_size:
            prototype_size = embeddings_size

        self.prototype_size = prototype_size

        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size, prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)
        else:
            self.metric_space_decoder = None

        # create initial prototypes for all classes
        # (all initial prototypes are a vector of all 1s)
        self.prototype_vectors = torch.nn.Parameter(
            torch.ones(num_prototypes, prototype_size), requires_grad=True)

        if use_radius:
            self.prototype_radii = torch.nn.Parameter(
                torch.ones(num_prototypes), requires_grad=True)
        else:
            self.prototype_radii = None

        # if set, create initial prototypes from normal distribution
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(
                torch.normal(torch.zeros(num_prototypes, prototype_size)))

        self.min_radius = min_radius
        self.learning_mode = learning_mode

        assert (unlabeled_idx is None) == (unlabeled_distance is None), (
            "'unlabeled_idx' and 'unlabeled_distance' should either both be set or both not be set.")

        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_distance = unlabeled_distance

        self._distance_function = distance_function

        if distance_function.lower() == 'hyperbolic':
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == 'cosine':
            self.distance = CosineDistance()
        elif distance_function.lower() == 'logit_cosine':
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == 'euclidean':
            self.distance = EuclideanDistance()
        elif distance_function.lower() == 'dot_product':
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f'Distance function {distance_function} not found.')


        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded):
        if self.learning_mode == 'learn_only_map_and_prototypes':
            embedded = embedded.detach()

        # decode embeddings into prototype space
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded

        prot = self.prototype_vectors
        radii = self.prototype_radii

        if self.learning_mode == 'learn_only_prototypes':
            encoded = encoded.detach()

        if self.learning_mode == 'learn_only_embeddings_and_map':
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

    def enable_expectation_maximization(self,
                                        data: FlairDataset,
                                        encoder: DefaultClassifier,
                                        exempt_labels : Optional[List[str]] = [],
                                        mini_batch_size : int = 8):
        """Applies monkey-patch to train method (which sets the train flag).

        This allows for computation of average prototypes after a training
        sequence. """

        decoder = self

        unpatched_train = encoder.train

        def patched_train(mode: bool = True):
            unpatched_train(mode=mode)
            if mode:
                logger.info("recalculating prototypes")
                with torch.no_grad():
                    decoder.calculate_prototypes(
                        data=data, encoder=encoder, exempt_labels=exempt_labels,
                        mini_batch_size=mini_batch_size)

        encoder.train = patched_train

    def calculate_prototypes(self, data: FlairDataset, encoder: DefaultClassifier, exempt_labels : Optional[List[str]] = [], mini_batch_size=32):
        """
        Function that calclues a prototype for each class based on the euclidean average embedding over the whole dataset
        :param dataset: dataset for which to calculate prototypes
        :param mini_batch_size: number of sentences to embed at same time
        :return:
        """

        dataloader = DataLoader(data, batch_size=mini_batch_size)

        # reset prototypes for all classes
        new_prototypes = torch.zeros(self.num_prototypes, self.prototype_size, device=flair.device)

        counter = Counter()

        for batch in tqdm(dataloader):

            logits, labels = encoder.forward_pass(batch)

            # decode embeddings into prototype space
            if self.metric_space_decoder is not None:
                logits = self.metric_space_decoder(logits)

            for logit, label in zip(logits, labels):
                counter.update(label)

                idx = encoder.label_dictionary.get_idx_for_item(label[0])

                new_prototypes[idx] += logit

        # TODO: changes required
        for label, count in counter.most_common():
            average_prototype = new_prototypes[encoder.label_dictionary.get_idx_for_item(label)] / count
            new_prototypes[encoder.label_dictionary.get_idx_for_item(label)] = average_prototype

        for label in exempt_labels:
            label_idx = encoder.label_dictionary.get_idx_for_item(label)
            new_prototypes[label_idx] = self.prototype_vectors[label_idx]

        self.prototype_vectors.data = new_prototypes.detach().to(flair.device)
