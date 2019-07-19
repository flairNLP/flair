import itertools

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
        # self.modality_b_classifier = nn.Linear(self.modality_b_embeddings.embedding_length, 1)

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
        # self.modality_b_classifier.weigths = batch_modality_b_classifier_parameters[:-1]
        # self.modality_b_classifier.bias = batch_modality_b_classifier_parameters[1]
        # batch_modality_b_logits = self.modality_b_classifier.forward(batch_modality_b_embeddings)  <== this needs to go for both positive and negative classifiers

        n_b = batch_modality_b_embeddings.shape[0]
        ones = torch.ones(n_b, 1).to(flair.device)
        batch_modality_b_embeddings = torch.cat([batch_modality_b_embeddings, ones], dim=-1)
        batch_modality_b_logits = torch.matmul(batch_modality_b_classifier_parameters, batch_modality_b_embeddings.t())
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