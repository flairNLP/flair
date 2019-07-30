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

from typing import Union, List
from pathlib import Path


# == similarity net ==
class SimilarityNet(flair.nn.Model):
    """
    A model that encapsulates modality specific models and a model shared between both modalities.
    Modality specific models are aligning the representations into a shared space, and modality shared model is aligning
    further these shared modalities.
    """

    def __init__(
        self, modality_specific_0=None, modality_specific_1=None, modality_shared=None
    ):
        super(SimilarityNet, self).__init__()

        # replace any None with identity (pass-through) model
        def embed_or_identity(x):
            return x if x is not None else nn.Identity()

        self.modality_specific_0 = embed_or_identity(modality_specific_0)
        self.modality_specific_1 = embed_or_identity(modality_specific_1)
        self.modality_shared = embed_or_identity(modality_shared)

    def forward(self, x):
        input_modality_0 = x[0]
        input_modality_1 = x[1]
        # skip the modalities which are not available (None)
        return [
            self.modality_shared(self.modality_specific_0(input_modality_0))
            if input_modality_0 is not None
            else None,
            self.modality_shared(self.modality_specific_1(input_modality_1))
            if input_modality_1 is not None
            else None,
        ]


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


class RBFSimilarity(SimilarityMeasure):
    """
    Similarity defined by radial basis function with a given bandwidth.
    """

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def forward(self, x):
        input_modality_0 = x[0]
        input_modality_1 = x[1]

        input_modality_0_sqnorms = torch.sum(
            input_modality_0 ** 2, dim=-1, keepdim=True
        )
        input_modality_1_sqnorms = torch.sum(
            input_modality_1 ** 2, dim=-1, keepdim=True
        )

        neg_sq_dist = 2 * torch.matmul(input_modality_0, input_modality_1.t())
        neg_sq_dist -= input_modality_0_sqnorms
        neg_sq_dist -= input_modality_1_sqnorms.t()

        return torch.exp(neg_sq_dist / self.bandwidth)


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


class sqL2Similarity(SimilarityMeasure):
    """
    The similarity defined by negative squared L2 distance.
    """

    def forward(self, x):
        # this returns *negative* squared L2 distance

        input_modality_0 = x[0]
        input_modality_1 = x[1]

        input_modality_0_sqnorms = torch.sum(
            input_modality_0 ** 2, dim=-1, keepdim=True
        )
        input_modality_1_sqnorms = torch.sum(
            input_modality_1 ** 2, dim=-1, keepdim=True
        )

        return (
            2 * torch.matmul(input_modality_0, input_modality_1.t())
            - input_modality_0_sqnorms
            - input_modality_1_sqnorms.t()
        )


# == similarity losses ==
class SimilarityLoss(nn.modules.loss._Loss):
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
        input_modality_0_embedding: Embeddings,
        input_modality_1_embedding: Embeddings,
        similarity_net: SimilarityNet,
        similarity_measure: SimilarityMeasure,
        similarity_loss: SimilarityLoss,
        eval_device=flair.device,
        recall_at_points=[1, 5, 10, 20],
        recall_at_points_weights=[0.4, 0.3, 0.2, 0.1]
    ):
        super(SimilarityLearner, self).__init__()
        self.input_modality_0_embedding = input_modality_0_embedding
        self.input_modality_1_embedding = input_modality_1_embedding
        self.similarity_net = similarity_net
        self.similarity_measure = similarity_measure
        self.similarity_loss = similarity_loss
        self.eval_device = eval_device
        self.recall_at_points = recall_at_points
        self.recall_at_points_weights = recall_at_points_weights
        self.to(flair.device)

    def _embed_inputs(self, data_points, modality_ids=[0, 1]):
        modality_ids = (
            modality_ids if isinstance(modality_ids, list) else [modality_ids]
        )
        sample_pairs = True if len(modality_ids) == 2 else False
        # extracts modality embeddings from each data point and puts them into separate tensors
        modality_tensors = [[], []]
        pair_indices = []
        for point_id, point in enumerate(data_points):
            if sample_pairs:
                if point.data[0] is not None and point.data[1] is not None:
                    pair_index = []
                    for modality_id in modality_ids:
                        index = np.random.choice(len(point.data[modality_id]))
                        modality_embedding = (
                            self.input_modality_0_embedding
                            if modality_id == 0
                            else self.input_modality_1_embedding
                        )
                        modality_embedding.embed(point.data[modality_id][index])
                        modality_tensors[modality_id].append(
                            point.data[modality_id][index].embedding.to(flair.device)
                        )
                        pair_index.append((point_id, index))
                    pair_indices.append(pair_index)
            else:
                assert len(modality_ids) == 1
                modality_id = modality_ids[0]
                assert modality_id in [0, 1]
                modality_embedding = (
                    self.input_modality_0_embedding
                    if modality_id == 0
                    else self.input_modality_1_embedding
                )
                modality_data = point.data[modality_id]
                if modality_data is not None:
                    modality_embedding.embed(modality_data)
                    modality_tensors[modality_id].extend(
                        [d.embedding.to(flair.device) for d in modality_data]
                    )
                    first = [(point_id, index) for index in range(len(modality_data))]
                    second = len(modality_data) * [None]
                    first, second = (
                        (second, first) if modality_id == 1 else (first, second)
                    )
                    pair_indices.extend(list(zip(first, second)))

        # store_embeddings(data_points, 'none')

        output_modality_tensors = []
        for modality_tensor in modality_tensors:
            if modality_tensor:
                output_modality_tensors.append(torch.stack(modality_tensor))
            else:
                output_modality_tensors.append(None)

        return output_modality_tensors, pair_indices

    def embed(
        self, data_points: Union[List[DataPoint], DataPoint], modality_id
    ) -> List[DataPoint]:
        data_points_in = []
        if isinstance(data_points, list):
            if not isinstance(data_points[0], DataPair):
                for data in data_points:
                    d = data if isinstance(data, list) else [data]
                    data_points_in.append(DataPair([d, None]) if modality_id==0 else DataPair([None, d]))
        else:
            if not isinstance(data_points, DataPair):
                d = data_points if isinstance(data_points, list) else [data_points]
                data_points_in = [DataPair([d, None] if modality_id==0 else [None, d])]
            else:
                data_points_in = [data_points]

        embedded_inputs, _ = self._embed_inputs(data_points_in, modality_ids=[modality_id])
        aligned_embeddings = self.similarity_net.forward(embedded_inputs)

        return aligned_embeddings[modality_id]

    def get_similarity(self, modality_0_embeddings, modality_1_embeddings):
        """
        :param modality_0_embedding: embeddings of first modality, a tensor of shape [n0, d0]
        :param modality_1_embeddings: embeddings of second modality, a tensor of shape [n1, d1]
        :return: a similarity matrix of shape [n0, n1]
        """
        return self.similarity_measure.forward([modality_0_embeddings, modality_1_embeddings])

    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        embedded_inputs, _ = self._embed_inputs(data_points, modality_ids=[0, 1])
        aligned_embeddings = self.similarity_net.forward(embedded_inputs)
        similarity_matrix = self.similarity_measure.forward(aligned_embeddings)
        targets = torch.eye(similarity_matrix.shape[0]).to(flair.device)
        loss = self.similarity_loss.forward(similarity_matrix, targets)

        return loss

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embeddings_storage_mode="cpu",
    ) -> (Result, float):
        # assumes that for each data pair there's at least one embedding per modality

        # embed_inputs on modality 1
        modality_1_embeddings = []
        for data_points in data_loader:
            embedded_inputs, _ = self._embed_inputs(data_points, modality_ids=[1])
            batch_embeddings = self.similarity_net.forward(embedded_inputs)
            batch_modality_1_embeddings = batch_embeddings[1]
            modality_1_embeddings.append(
                batch_modality_1_embeddings.to(self.eval_device)
            )
            store_embeddings(data_points, embeddings_storage_mode)
        modality_1_embeddings = torch.cat(modality_1_embeddings, dim=0)  # [n0, d0]

        ranks = []
        modality_1_id = 0
        for data_points in data_loader:
            # embed_inputs on modality 0
            embedded_inputs, indices = self._embed_inputs(data_points, modality_ids=[0])
            modality_1_indices = modality_1_id + torch.LongTensor(
                [index[0][0] for index in indices]
            )
            batch_embeddings = self.similarity_net.forward(embedded_inputs)
            batch_modality_0_embeddings = batch_embeddings[0]
            batch_modality_0_embeddings = batch_modality_0_embeddings.to(
                self.eval_device
            )
            # compute the similarity
            batch_similarity_matrix = self.similarity_measure.forward(
                [batch_modality_0_embeddings, modality_1_embeddings]
            )  # [bn_1, n0]
            batch_modality_1_argsort = torch.argsort(
                batch_similarity_matrix, descending=True, dim=1
            )
            # get the ranks, so +1 to start counting ranks from 1
            batch_modality_1_ranks = torch.argsort(batch_modality_1_argsort, dim=1) + 1
            batch_gt_ranks = batch_modality_1_ranks[
                torch.arange(batch_similarity_matrix.shape[0]), modality_1_indices
            ]
            ranks.extend(batch_gt_ranks.tolist())
            modality_1_id += len(data_points)
            store_embeddings(data_points, embeddings_storage_mode)

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
            "input_modality_0_embedding": self.input_modality_0_embedding,
            "input_modality_1_embedding": self.input_modality_1_embedding,
            "similarity_net": self.similarity_net,
            "similarity_measure": self.similarity_measure,
            "similarity_loss": self.similarity_loss,
            "eval_device": self.eval_device,
            "recall_at_points": self.recall_at_points,
            "recall_at_points_weights": self.recall_at_points_weights,
        }
        return model_state

    def _init_model_with_state_dict(state):
        # The conversion from old model's constructor interface
        if "input_embeddings" in state:
            state["input_modality_0_embedding"] = state["input_embeddings"][0]
            state["input_modality_1_embedding"] = state["input_embeddings"][1]
        model = SimilarityLearner(
            input_modality_0_embedding=state["input_modality_0_embedding"],
            input_modality_1_embedding=state["input_modality_1_embedding"],
            similarity_net=state["similarity_net"],
            similarity_measure=state["similarity_measure"],
            similarity_loss=state["similarity_loss"],
            eval_device=state["eval_device"],
            recall_at_points=state["recall_at_points"],
            recall_at_points_weights=state["recall_at_points_weights"],
        )

        model.load_state_dict(state["state_dict"])
        return model
