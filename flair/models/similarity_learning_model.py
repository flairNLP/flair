from abc import abstractmethod

import flair
from flair.data import DataPoint, DataPair
from flair.embeddings import Embeddings
from flair.datasets import DataLoader
from flair.training_utils import Result
from flair.training_utils import store_embeddings

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import itertools

from typing import Union, List
from pathlib import Path


# == similarity measures ==
class SimilarityMeasure:
    @abstractmethod
    def forward(self, x):
        pass


# helper class for ModelSimilarity
class SliceReshaper(flair.nn.Model):
    def __init__(self, begin, end=None, shape=None):
        super(SliceReshaper, self).__init__()
        self.begin = begin
        self.end = end
        self.shape = shape

    def forward(self, x):
        x = x[:, self.begin] if self.end is None else x[:, self.begin : self.end]
        x = x.view(-1, *self.shape) if self.shape is not None else x
        return x


# -- works with binary cross entropy loss --
class ModelSimilarity(SimilarityMeasure):
    """
    Similarity defined by the model. The model parameters are given by the first element of the pair.
    The similarity is evaluated by doing the forward pass (inference) on the parametrized model with
    the second element of the pair as input.
    """

    def __init__(self, model):
        # model is a list of tuples (function, parameters), where parameters is a dict {param_name: param_extract_model}
        self.model = model

    def forward(self, x):

        model_parameters = x[0]
        model_inputs = x[1]

        cur_outputs = model_inputs
        for layer_model, parameter_map in self.model:
            param_dict = {}
            for param_name, param_slice_reshape in parameter_map.items():
                if isinstance(param_slice_reshape, SliceReshaper):
                    val = param_slice_reshape(model_parameters)
                else:
                    val = param_slice_reshape
                param_dict[param_name] = val
            cur_outputs = layer_model(cur_outputs, **param_dict)

        return cur_outputs


# -- works with ranking/triplet loss --
class CosineSimilarity(SimilarityMeasure):
    """
    Similarity defined by the cosine distance.
    """

    def forward(self, x):
        input_modality_0 = x[0]
        input_modality_1 = x[1]

        # normalize the embeddings
        input_modality_0_norms = torch.norm(input_modality_0, dim=-1, keepdim=True)
        input_modality_1_norms = torch.norm(input_modality_1, dim=-1, keepdim=True)

        return torch.matmul(
            input_modality_0 / input_modality_0_norms,
            (input_modality_1 / input_modality_1_norms).t(),
        )


# == similarity losses ==
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class PairwiseBCELoss(SimilarityLoss):
    """
    Binary cross entropy between pair similarities and pair labels.
    """

    def __init__(self, balanced=False):
        super(PairwiseBCELoss, self).__init__()
        self.balanced = balanced

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets).to(flair.device) - targets
        # we want that logits for corresponding pairs are high, and for non-corresponding low
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        if self.balanced:
            # TODO: this assumes eye matrix
            weight_matrix = n * (targets / 2.0 + neg_targets / (2.0 * (n - 1)))
            bce_loss *= weight_matrix
        loss = bce_loss.mean()

        return loss


class RankingLoss(SimilarityLoss):
    """
    Triplet ranking loss between pair similarities and pair labels.
    """

    def __init__(self, margin=0.1, direction_weights=[0.5, 0.5]):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.direction_weights = direction_weights

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        # loss matrices for two directions of alignment, from modality 0 => modality 1 and vice versa
        ranking_loss_matrix_01 = neg_targets * F.relu(
            self.margin + inputs - torch.diag(inputs).view(n, 1)
        )
        ranking_loss_matrix_10 = neg_targets * F.relu(
            self.margin + inputs - torch.diag(inputs).view(1, n)
        )
        neg_targets_01_sum = torch.sum(neg_targets, dim=1)
        neg_targets_10_sum = torch.sum(neg_targets, dim=0)
        loss = self.direction_weights[0] * torch.mean(
            torch.sum(ranking_loss_matrix_01 / neg_targets_01_sum, dim=1)
        ) + self.direction_weights[1] * torch.mean(
            torch.sum(ranking_loss_matrix_10 / neg_targets_10_sum, dim=0)
        )

        return loss


# == similarity learner ==
class SimilarityLearner(flair.nn.Model):
    def __init__(
        self,
        source_embeddings: Embeddings,
        target_embeddings: Embeddings,
        similarity_measure: SimilarityMeasure,
        similarity_loss: SimilarityLoss,
        eval_device=flair.device,
        source_mapping: torch.nn.Module = None,
        target_mapping: torch.nn.Module = None,
        recall_at_points: List[int] = [1, 5, 10, 20],
        recall_at_points_weights: List[float] = [0.4, 0.3, 0.2, 0.1],
        interleave_embedding_updates: bool = False,
    ):
        super(SimilarityLearner, self).__init__()
        self.source_embeddings: Embeddings = source_embeddings
        self.target_embeddings: Embeddings = target_embeddings
        self.source_mapping: torch.nn.Module = source_mapping
        self.target_mapping: torch.nn.Module = target_mapping
        self.similarity_measure: SimilarityMeasure = similarity_measure
        self.similarity_loss: SimilarityLoss = similarity_loss
        self.eval_device = eval_device
        self.recall_at_points: List[int] = recall_at_points
        self.recall_at_points_weights: List[float] = recall_at_points_weights
        self.interleave_embedding_updates = interleave_embedding_updates

        self.to(flair.device)

    def _embed_source(self, data_points):

        if type(data_points[0]) == DataPair:
            data_points = [point.first for point in data_points]

        self.source_embeddings.embed(data_points)

        source_embedding_tensor = torch.stack(
            [point.embedding for point in data_points]
        ).to(flair.device)

        if self.source_mapping is not None:
            source_embedding_tensor = self.source_mapping(source_embedding_tensor)

        return source_embedding_tensor

    def _embed_target(self, data_points):

        if type(data_points[0]) == DataPair:
            data_points = [point.second for point in data_points]

        self.target_embeddings.embed(data_points)

        target_embedding_tensor = torch.stack(
            [point.embedding for point in data_points]
        ).to(flair.device)

        if self.target_mapping is not None:
            target_embedding_tensor = self.target_mapping(target_embedding_tensor)

        return target_embedding_tensor

    def get_similarity(self, modality_0_embeddings, modality_1_embeddings):
        """
        :param modality_0_embeddings: embeddings of first modality, a tensor of shape [n0, d0]
        :param modality_1_embeddings: embeddings of second modality, a tensor of shape [n1, d1]
        :return: a similarity matrix of shape [n0, n1]
        """
        return self.similarity_measure.forward(
            [modality_0_embeddings, modality_1_embeddings]
        )

    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        mapped_source_embeddings = self._embed_source(data_points)
        mapped_target_embeddings = self._embed_target(data_points)

        if self.interleave_embedding_updates:
            # 1/3 only source branch of model, 1/3 only target branch of model, 1/3 both
            detach_modality_id = torch.randint(0, 3, (1,)).item()
            if detach_modality_id == 0:
                mapped_source_embeddings.detach()
            elif detach_modality_id == 1:
                mapped_target_embeddings.detach()

        similarity_matrix = self.similarity_measure.forward(
            (mapped_source_embeddings, mapped_target_embeddings)
        )

        def add_to_index_map(hashmap, key, val):
            if key not in hashmap:
                hashmap[key] = [val]
            else:
                hashmap[key] += [val]

        index_map = {"first": {}, "second": {}}
        for data_point_id, data_point in enumerate(data_points):
            add_to_index_map(index_map["first"], str(data_point.first), data_point_id)
            add_to_index_map(index_map["second"], str(data_point.second), data_point_id)

        targets = torch.zeros_like(similarity_matrix).to(flair.device)

        for data_point in data_points:
            first_indices = index_map["first"][str(data_point.first)]
            second_indices = index_map["second"][str(data_point.second)]
            for first_index, second_index in itertools.product(
                first_indices, second_indices
            ):
                targets[first_index, second_index] = 1.0

        loss = self.similarity_loss(similarity_matrix, targets)

        return loss

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode="none",
    ) -> (Result, float):
        # assumes that for each data pair there's at least one embedding per modality

        with torch.no_grad():
            # pre-compute embeddings for all targets in evaluation dataset
            target_index = {}
            all_target_embeddings = []
            for data_points in data_loader:
                target_inputs = []
                for data_point in data_points:
                    if str(data_point.second) not in target_index:
                        target_index[str(data_point.second)] = len(target_index)
                        target_inputs.append(data_point)
                if target_inputs:
                    all_target_embeddings.append(
                        self._embed_target(target_inputs).to(self.eval_device)
                    )
                store_embeddings(data_points, embedding_storage_mode)
            all_target_embeddings = torch.cat(all_target_embeddings, dim=0)  # [n0, d0]
            assert len(target_index) == all_target_embeddings.shape[0]

            ranks = []
            for data_points in data_loader:
                batch_embeddings = self._embed_source(data_points)

                batch_source_embeddings = batch_embeddings.to(self.eval_device)
                # compute the similarity
                batch_similarity_matrix = self.similarity_measure.forward(
                    [batch_source_embeddings, all_target_embeddings]
                )

                # sort the similarity matrix across modality 1
                batch_modality_1_argsort = torch.argsort(
                    batch_similarity_matrix, descending=True, dim=1
                )

                # get the ranks, so +1 to start counting ranks from 1
                batch_modality_1_ranks = (
                    torch.argsort(batch_modality_1_argsort, dim=1) + 1
                )

                batch_target_indices = [
                    target_index[str(data_point.second)] for data_point in data_points
                ]

                batch_gt_ranks = batch_modality_1_ranks[
                    torch.arange(batch_similarity_matrix.shape[0]),
                    torch.tensor(batch_target_indices),
                ]
                ranks.extend(batch_gt_ranks.tolist())

                store_embeddings(data_points, embedding_storage_mode)

        ranks = np.array(ranks)
        median_rank = np.median(ranks)
        recall_at = {k: np.mean(ranks <= k) for k in self.recall_at_points}

        results_header = ["Median rank"] + [
            "Recall@top" + str(r) for r in self.recall_at_points
        ]
        results_header_str = "\t".join(results_header)
        epoch_results = [str(median_rank)] + [
            str(recall_at[k]) for k in self.recall_at_points
        ]
        epoch_results_str = "\t".join(epoch_results)
        detailed_results = ", ".join(
            [f"{h}={v}" for h, v in zip(results_header, epoch_results)]
        )

        validated_measure = sum(
            [
                recall_at[r] * w
                for r, w in zip(self.recall_at_points, self.recall_at_points_weights)
            ]
        )

        return (
            Result(
                validated_measure,
                results_header_str,
                epoch_results_str,
                detailed_results,
            ),
            0,
        )

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "input_modality_0_embedding": self.source_embeddings,
            "input_modality_1_embedding": self.target_embeddings,
            "similarity_measure": self.similarity_measure,
            "similarity_loss": self.similarity_loss,
            "source_mapping": self.source_mapping,
            "target_mapping": self.target_mapping,
            "eval_device": self.eval_device,
            "recall_at_points": self.recall_at_points,
            "recall_at_points_weights": self.recall_at_points_weights,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        # The conversion from old model's constructor interface
        if "input_embeddings" in state:
            state["input_modality_0_embedding"] = state["input_embeddings"][0]
            state["input_modality_1_embedding"] = state["input_embeddings"][1]
        model = SimilarityLearner(
            source_embeddings=state["input_modality_0_embedding"],
            target_embeddings=state["input_modality_1_embedding"],
            source_mapping=state["source_mapping"],
            target_mapping=state["target_mapping"],
            similarity_measure=state["similarity_measure"],
            similarity_loss=state["similarity_loss"],
            eval_device=state["eval_device"],
            recall_at_points=state["recall_at_points"],
            recall_at_points_weights=state["recall_at_points_weights"],
        )

        model.load_state_dict(state["state_dict"])
        return model
