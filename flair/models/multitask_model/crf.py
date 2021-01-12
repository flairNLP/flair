import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import flair
from flair.data import Dictionary
from flair.models.sequence_tagger_model import log_sum_exp_batch

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

class CRF(nn.Module):

    def __init__(self, hidden_dim, tag_dictionary, tagset_size):
        super(CRF, self).__init__()
        self.emission = nn.Linear(hidden_dim, tagset_size)
        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.detach()[tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
        self.transitions.detach()[:, tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

    def forward_alg(self, features, lengths):
        emission_scores = self.emission(features)

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            emission_scores.shape[0],
            emission_scores.shape[1] + 1,
            emission_scores.shape[2],
            dtype=torch.float,
            device=flair.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(emission_scores.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(emission_scores.shape[0], 1, 1)

        for i in range(emission_scores.shape[1]):
            emit_score = emission_scores[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lengths, :]

        terminal_var = forward_var + self.transitions[
                                         self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def score_sentence(self, features, tags, lengths):
        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lengths)):
            pad_stop_tags[i, lengths[i]:] = self.tag_dictionary.get_idx_for_item(
                STOP_TAG
            )

        score = torch.FloatTensor(features.shape[0]).to(flair.device)

        for i in range(features.shape[0]):
            r = torch.LongTensor(range(lengths[i])).to(flair.device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lengths[i] + 1], pad_start_tags[i, : lengths[i] + 1]
                ]
            ) + torch.sum(features[i, r, tags[i, : lengths[i]]])

        return score