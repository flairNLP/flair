from abc import abstractmethod

import flair
from flair.data import DataPoint
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

    def __init__(self, modality_specific, modality_shared=None):
        super(SimilarityNet, self).__init__()
        # replace any None with pass-through net
        self.modality_shared     = modality_shared if modality_shared is not None else nn.Identity()
        self.modality_0_specific = modality_specific[0] if modality_specific[0] is not None else nn.Identity()
        self.modality_1_specific = modality_specific[1] if modality_specific[1] is not None else nn.Identity()

    def forward(self, x):
        # skip the modalities which are not available (None)
        return [self.modality_shared(self.modality_0_specific(x[0])) if x[0] is not None else None,
                self.modality_shared(self.modality_1_specific(x[1])) if x[1] is not None else None]


# == similarity measures ==
class SimilarityMeasure:

    @abstractmethod
    def forward(self):
        pass


class LogitSimilarity(SimilarityMeasure):

    def __init__(self, dynamic_model):
        self.dynamic_model = dynamic_model

    def forward(self, aligned_embeddings):
        # dynamic_model is a list of tuples (functional, parameters), where parameters is a dict {param_name: range}
        model_parameters = aligned_embeddings[0]
        model_inputs = aligned_embeddings[1]

        cur_outputs = model_inputs
        for layer_model, parameter_map in self.dynamic_model:
            param_dict = {param_name: param_reshape_func(model_parameters) for param_name, param_reshape_func in parameter_map.items()}
            cur_outputs = layer_model(cur_outputs, **param_dict)

        # transpose so the output is of dimensions [n_embeddings_0, n_embeddings_1]
        # cur_outputs = cur_outputs.t()

        return cur_outputs


# class RBFSimilarity(SimilarityMeasure):
#
#     def forward(self, aligned_embeddings):
#         pass


# -- works with ranking/triplet losses --
class CosineSimilarity(SimilarityMeasure):

    def forward(self, aligned_embeddings):
        aligned_embeddings_n = [aligned_embedding / torch.norm(aligned_embedding, dim=-1, keepdim=True) for aligned_embedding in aligned_embeddings]

        return torch.matmul(aligned_embeddings_n[0], aligned_embeddings_n[1].t())


class sqL2Similarity(SimilarityMeasure):

    def forward(self, aligned_embeddings):
        # this returns *negative* squared L2 distance
        norms = [torch.norm(aligned_embedding, dim=-1, keepdim=True) for aligned_embedding in aligned_embeddings]

        return 2 * torch.matmul(aligned_embeddings[0], aligned_embeddings[1].t()) - norms[0] - norms[1].t()


# == similarity losses ==
class SimilarityLoss(nn.modules.loss._Loss):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass

class PairwiseBCELoss(SimilarityLoss):

    def __init__(self, balanced=False):
        super(PairwiseBCELoss, self).__init__()
        self.balanced = balanced

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets).to(flair.device) - targets
        # we want that logits for corresponding pairs are high, and for non-corresponding low
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.balanced:
            # TODO: this assumes only eye matrix
            weight_matrix = n * (targets / 2. + neg_targets / (2. * (n-1)))
            bce_loss *= weight_matrix
        loss = bce_loss.mean()

        return loss


class RankingLoss(SimilarityLoss):

    def __init__(self, margin=0.1, direction_weights=[0.5, 0.5]):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.direction_weights = direction_weights

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        ranking_loss_matrix_modality_1to2 = neg_targets * F.relu(self.margin + inputs - torch.diag(inputs).view(n, 1))
        ranking_loss_matrix_modality_2to1 = neg_targets * F.relu(self.margin + inputs - torch.diag(inputs).view(1, n))
        neg_targets_1to2_sum = torch.sum(neg_targets, dim=1)
        neg_targets_2to1_sum = torch.sum(neg_targets, dim=0)
        loss = self.direction_weights[0] + torch.mean(neg_targets * ranking_loss_matrix_modality_1to2 / neg_targets_1to2_sum) + \
               self.direction_weights[1] + torch.mean(neg_targets * ranking_loss_matrix_modality_2to1 / neg_targets_2to1_sum)

        return loss


# == similarity learner ==
class SimilarityLearner_new(flair.nn.Model):

    def __init__(self, input_embeddings, similarity_net, similarity_measure, similarity_loss):
        super(SimilarityLearner_new, self).__init__()
        self.input_modality_0_embedding = input_embeddings[0]
        self.input_modality_1_embedding = input_embeddings[1]
        self.similarity_net = similarity_net
        self.similarity_measure = similarity_measure
        self.similarity_loss = similarity_loss
        self.to(flair.device)

    def _embed_inputs(self, data_points, modality_ids=[0,1]):
        modality_ids = modality_ids if isinstance(modality_ids, list) else [modality_ids]
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
                        modality_embedding = self.input_modality_0_embedding if modality_id == 0 else self.input_modality_1_embedding
                        modality_embedding.embed(point.data[modality_id][index])
                        modality_tensors[modality_id].append(point.data[modality_id][index].embedding.to(flair.device))
                        pair_index.append((point_id, index))
                    pair_indices.append(pair_index)
            else:
                assert(len(modality_ids) == 1)
                modality_id = modality_ids[0]
                modality_embedding = self.input_modality_0_embedding if modality_id == 0 else self.input_modality_1_embedding
                assert(modality_id <= 1 and modality_id >= 0)
                modality_data = point.data[modality_id]
                if modality_data is not None:
                    modality_embedding.embed(modality_data)
                    modality_tensors[modality_id].extend([d.embedding.to(flair.device) for d in modality_data])
                    first = [(point_id, index) for index in range(len(modality_data))]
                    second = len(modality_data) * [None]
                    first, second = (second, first) if modality_id == 1 else (first, second)
                    pair_indices.extend(list(zip(first, second)))

        # store_embeddings(data_points, 'none')

        output_modality_tensors = []
        for modality_tensor in modality_tensors:
            if modality_tensor:
                output_modality_tensors.append(torch.stack(modality_tensor))
            else:
                output_modality_tensors.append(None)

        return output_modality_tensors, pair_indices

    def embed(self, data_points: Union[List[DataPoint], DataPoint], modality_id) -> List[DataPoint]:
        embedded_inputs, _ = self._embed_inputs(data_points, modality_ids=[modality_id])
        aligned_embeddings = self.similairty_net.forward(embedded_inputs)

        return aligned_embeddings[modality_id]

    def forward_loss(self, data_points: Union[List[DataPoint], DataPoint]) -> torch.tensor:
        embedded_inputs, _ = self._embed_inputs(data_points, modality_ids=[0,1])
        aligned_embeddings = self.similarity_net.forward(embedded_inputs)
        similarity_matrix = self.similarity_measure.forward(aligned_embeddings)
        targets = torch.eye(similarity_matrix.shape[0]).to(flair.device)
        loss = self.similarity_loss.forward(similarity_matrix, targets)

        return loss

    def evaluate(self, data_loader: DataLoader, out_path: Path = None) -> (Result, float):
        # this assumes that data_loader does not do shuffling,
        # that each modality 0 has at least one modality 1 and vice versa

        eval_device = flair.device

        # embed_inputs on modality 1
        modality_1_embeddings = []
        modality_id = 1
        for data_points in data_loader:
            embedded_inputs, _ = self._embed_inputs(data_points, modality_ids=[modality_id])
            modality_1_embeddings.append(self.similarity_net.forward(embedded_inputs)[modality_id].to(eval_device))
            store_embeddings(data_points, 'none')
        modality_1_embeddings = torch.cat(modality_1_embeddings, dim=0) # [n0, d0]

        ranks = []
        modality_id = 0
        modality_1_id = 0
        for data_points in data_loader:
            # embed_inputs on modality 0
            embedded_inputs, indices = self._embed_inputs(data_points, modality_ids=[modality_id])
            batch_modality_1_indices = np.array([index[0][0] for index in indices])
            modality_1_indices = modality_1_id + batch_modality_1_indices
            # DEBUG: print(modality_1_indices)
            batch_modality_0_embeddings = self.similarity_net.forward(embedded_inputs)[modality_id].to(eval_device)  # [bn_1, d]
            # compute the similarity
            batch_similarity_matrix = self.similarity_measure.forward([batch_modality_0_embeddings, modality_1_embeddings]) # [bn_1, n0]
            batch_modality_1_argsort = torch.argsort(batch_similarity_matrix, descending=True, dim=1)
            # get the ranks, so +1 to start counting ranks from 1
            batch_modality_1_ranks = torch.argsort(batch_modality_1_argsort, dim=1) + 1
            batch_gt_ranks = batch_modality_1_ranks[torch.arange(batch_similarity_matrix.shape[0]), torch.LongTensor(modality_1_indices)]
            ranks.extend(batch_gt_ranks.tolist())
            modality_1_id += len(data_points)
            store_embeddings(data_points, 'none')

        recall_at_points = [1, 5, 10, 20]
        recall_at_points_weigths = [0.4, 0.3, 0.2, 0.1]

        ranks = np.array(ranks)
        median_rank = np.median(ranks)
        recall_at = {k: np.mean(ranks <= k) for k in recall_at_points}

        results_header = ['Median rank'] + ['Recall@top'+str(r) for r in recall_at_points]
        results_header_str = '\t'.join(results_header)
        epoch_results = [str(median_rank)] + [str(recall_at[k]) for k in recall_at_points]
        epoch_results_str = '\t'.join(epoch_results)
        detailed_results = ', '.join([f'{h}={v}' for h,v in zip(results_header, epoch_results)])

        validated_measure = sum([recall_at[r]*w for r,w in zip(recall_at_points, recall_at_points_weigths)])

        return (Result(validated_measure, results_header_str, epoch_results_str, detailed_results), 0)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "input_embeddings": [self.input_modality_0_embedding, self.input_modality_1_embedding],
            "similarity_net": self.similarity_net,
            "similarity_measure": self.similarity_measure,
            "similarity_loss": self.similarity_loss
        }
        return model_state

    def _init_model_with_state_dict(state):
        model = SimilarityLearner_new(
            input_embeddings=state["input_embeddings"],
            similarity_net=state["similarity_net"],
            similarity_measure=state["similarity_measure"],
            similarity_loss=state["similarity_loss"]
        )

        model.load_state_dict(state["state_dict"])
        return model


# == old version ==
class PairwiseBCELoss(nn.modules.loss._Loss):

    def __init__(self, balanced=False):
        super(PairwiseBCELoss, self).__init__()
        self.balanced = balanced

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones(n,n).to(flair.device) - targets
        # we want that logits for corresponding pairs are high, and for non-corresponding low
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.balanced:
            weight_matrix = n * (targets / 2. + neg_targets / (2. * (n-1)))
            bce_loss *= weight_matrix
        loss = bce_loss.mean()

        return loss


class RankingLoss(nn.modules.loss._Loss):

    def __init__(self, margin=0.1):
        # TODO: direction weights, so one can be more interested in one direction of alignment than the other, or if one trusts more labels in one direction than in the other
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        # TODO: this assumes a) only one direction & b) eye matrix for inputs
        n = inputs.shape[0]
        neg_targets = torch.ones(n,n) - targets
        ranking_loss_matrix = F.relu(self.margin * neg_targets + inputs - torch.diag(inputs).view(n, 1))
        loss = torch.sum(neg_targets * ranking_loss_matrix) / (n * (n - 1))

        return loss


class SimilarityLearner(flair.nn.Model):

    def __init__(self, modality_a_embeddings, modality_b_embeddings, loss=PairwiseBCELoss(), modality_a_embedding_dropout=0., modality_b_embedding_dropout=0.): # output_dim, text_embedding_parms):
        super(SimilarityLearner, self).__init__()

        self.modality_a_embeddings = modality_a_embeddings
        self.modality_b_embeddings = modality_b_embeddings

        self.modality_a_embedding_dropout = modality_a_embedding_dropout
        self.modality_a_dropout = nn.Dropout(self.modality_a_embedding_dropout)

        self.modality_b_embedding_dropout = modality_b_embedding_dropout
        self.modality_b_dropout = nn.Dropout(self.modality_b_embedding_dropout)

        self.modality_b_classifier_parameter_model = nn.Sequential(nn.Linear(self.modality_a_embeddings.embedding_length, self.modality_b_embeddings.embedding_length + 1))

        self.loss = loss

        self.to(flair.device)

    def forward_loss(self, data_points: Union[List[DataPoint], DataPoint]) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""  # TODO: change this text

        # sample modality pairs
        modality_a_inputs = []
        modality_b_inputs = []
        for d in data_points:
            (modality_a_input, modality_b_input) = d.sample_tuple()
            modality_a_inputs.append(modality_a_input)
            modality_b_inputs.append(modality_b_input)

        self.modality_a_embeddings.embed(modality_a_inputs)
        batch_modality_a_embeddings = torch.stack([a.embedding.to(flair.device) for a in modality_a_inputs])
        batch_modality_a_embeddings = self.modality_a_dropout.forward(batch_modality_a_embeddings)

        self.modality_b_embeddings.embed(modality_b_inputs)
        batch_modality_b_embeddings = torch.stack([b.embedding.to(flair.device) for b in modality_b_inputs])
        batch_modality_b_embeddings = self.modality_a_dropout.forward(batch_modality_b_embeddings)

        batch_modality_b_classifier_parameters = self.modality_b_classifier_parameter_model.forward(batch_modality_a_embeddings)
        batch_modality_b_logits = F.linear(batch_modality_b_embeddings,
                                           weight=batch_modality_b_classifier_parameters[:,:-1],
                                           bias=batch_modality_b_classifier_parameters[:,-1])

        n_b = batch_modality_b_embeddings.shape[0]
        batch_modality_b_targets = torch.eye(n_b).to(flair.device)

        batch_loss = self.loss.forward(batch_modality_b_logits, batch_modality_b_targets)

        return batch_loss

    def predict(self, data_points: Union[List[DataPoint], DataPoint], mini_batch_size=32) -> List[DataPoint]:
        """Predicts the labels/tags for the given list of sentences. The labels/tags are added directly to the
        sentences. Implement this to enable prediction."""
        pass

    def evaluate(self, data_loader: DataLoader, out_path: Path = None) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation."""

        recall_at_points = [1, 5, 10, 20]
        recall_at_points_weigths = [0.4, 0.3, 0.2, 0.1]

        # embed all points of modality b
        modality_b_embeddings = []
        for data_points in data_loader:
            modality_b_inputs = [d.data_points(1)[0] for d in data_points]
            self.modality_b_embeddings.embed(modality_b_inputs)
            modality_b_embeddings.extend([b.embedding.to(flair.device) for b in modality_b_inputs])
            store_embeddings(data_points, 'none') # modality_b_inputs, 'none')
        modality_b_embeddings = torch.stack(modality_b_embeddings)
        ones = torch.ones(modality_b_embeddings.shape[0], 1).to(flair.device)
        modality_b_embeddings = torch.cat([modality_b_embeddings, ones], dim=-1).t()

        ranks = []
        modality_b_id = 0
        for data_points in data_loader:
            # embed batch points for modality a
            modality_a_inputs = []
            modality_b_indices = []
            # for each tuple in a batch
            for d in data_points:
                # get all modality_a inputs, grouped by modality_b inputs
                cur_modality_a_inputs = d.group_by(0,1)
                # go over modality_a inputs
                for i, cur_modality_a_input in enumerate(cur_modality_a_inputs):
                    modality_a_inputs.extend(cur_modality_a_input)
                    modality_b_indices.extend(len(cur_modality_a_input) * [modality_b_id+i])
                modality_b_id += len(cur_modality_a_inputs)
            self.modality_a_embeddings.embed(modality_a_inputs)
            batch_modality_a_embeddings = torch.stack([a.embedding.to(flair.device) for a in modality_a_inputs])
            n_batch = batch_modality_a_embeddings.shape[0]
            #
            batch_modality_b_classifier_parameters = self.modality_b_classifier_parameter_model.forward(batch_modality_a_embeddings)
            batch_modality_b_logits = torch.matmul(batch_modality_b_classifier_parameters, modality_b_embeddings)
            batch_modality_b_argsort = torch.argsort(batch_modality_b_logits, descending=True, dim=1)
            # get the ranks, so +1 to start counting ranks from 1
            batch_modality_b_ranks = torch.argsort(batch_modality_b_argsort, dim=1) + 1
            batch_gt_ranks = batch_modality_b_ranks[torch.arange(n_batch), torch.LongTensor(modality_b_indices)]
            ranks.extend(batch_gt_ranks.tolist())
            store_embeddings(data_points, 'none') # modality_a_inputs, 'none')


        ranks = np.array(ranks)
        median_rank = np.median(ranks)
        recall_at = {k: np.mean(ranks <= k) for k in recall_at_points}

        results_header = ['Median rank'] + ['Recall@top'+str(r) for r in recall_at_points]
        results_header_str = '\t'.join(results_header)
        epoch_results = [str(median_rank)] + [str(recall_at[k]) for k in recall_at_points]
        epoch_results_str = '\t'.join(epoch_results)
        detailed_results = ', '.join([f'{h}={v}' for h,v in zip(results_header, epoch_results)])

        validated_measure = sum([recall_at[r]*w for r,w in zip(recall_at_points, recall_at_points_weigths)])

        return (Result(validated_measure, results_header_str, epoch_results_str, detailed_results), 0)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "modality_a_embeddings": self.modality_a_embeddings,
            "modality_b_embeddings": self.modality_b_embeddings,
            "loss": self.loss,
            "modality_a_embedding_dropout": self.modality_a_embedding_dropout,
            "modality_b_embedding_dropout": self.modality_b_embedding_dropout
        }
        return model_state

    def _init_model_with_state_dict(state):
        model = SimilarityLearner(
            modality_a_embeddings=state["modality_a_embeddings"],
            modality_b_embeddings=state["modality_b_embeddings"],
            loss=state["loss"],
            modality_a_embedding_dropout=state["modality_a_embedding_dropout"],
            modality_b_embedding_dropout=state["modality_b_embedding_dropout"]
        )

        model.load_state_dict(state["state_dict"])
        return model