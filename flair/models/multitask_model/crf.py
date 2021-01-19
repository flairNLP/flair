import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import flair
from flair.data import Dictionary
from flair.models.sequence_tagger_model import log_sum_exp_batch, START_TAG, STOP_TAG

class CRF(nn.Module):

    def __init__(self, hidden_dim, tag_dictionary, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.tag_dictionary = tag_dictionary
        self.emission = nn.Linear(hidden_dim, tagset_size)
        self.transitions = torch.nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.detach()[:, tag_dictionary.get_idx_for_item(START_TAG)] = -10000
        self.transitions.detach()[tag_dictionary.get_idx_for_item(STOP_TAG), :] = -10000

    def forward(self, lstm_features):
        self.batch_size = lstm_features.size(0)
        self.seq_len = lstm_features.size(1)

        emission_scores = self.emission(lstm_features)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.seq_len, self.tagset_size, self.tagset_size)

        crf_scores = emission_scores + self.transitions.unsqueeze(0).unsqueeze(0)
        return crf_scores

    def forward_alg(self, features, lengths):

        scores_up_to_t = torch.zeros(
            self.batch_size,
            self.seq_len + 1,
            self.tagset_size,
            dtype=torch.float,
            device=flair.device,
        )

        start_mask = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        start_mask[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
        scores_up_to_t[:, 0, :] = start_mask.unsqueeze(0).repeat(self.batch_size, 1)

        for token_num in range(max(lengths)):
            crf_score = features[:, token_num, :]
            score_up_to_t = scores_up_to_t[:,token_num,:]

            tag_var = (
                    crf_score
                    + score_up_to_t.unsqueeze(2).repeat(1, 1, self.tagset_size)
            )

            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.unsqueeze(1).repeat(1, self.tagset_size, 1)

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=1))

            cloned = scores_up_to_t.clone()
            cloned[:, token_num + 1, :] = max_tag_var + agg_

            scores_up_to_t = cloned

        indices = lengths.unsqueeze(-1).repeat(1, self.tagset_size).unsqueeze(1)
        scores_up_to_t = torch.gather(scores_up_to_t, 1, indices).squeeze(1)

        scores_up_to_t = scores_up_to_t + self.transitions[
                                            :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                        ].unsqueeze(0).repeat(self.batch_size, 1)

        all_paths_scores = log_sum_exp_batch(scores_up_to_t)

        return all_paths_scores

    def gold_score(self, crf_scores, tags, lengths):
        start_tag = torch.LongTensor([self.tag_dictionary.get_idx_for_item(START_TAG)])
        stop_tag = torch.LongTensor([self.tag_dictionary.get_idx_for_item(STOP_TAG)])

        gold_score = torch.FloatTensor(self.batch_size).to(flair.device)

        for batch_no in range(self.batch_size):
            token_indices = torch.arange(lengths[batch_no]).to(flair.device)
            gold_tags = tags[batch_no, : lengths[batch_no]]
            from_tags = torch.cat([start_tag, gold_tags])
            to_tags = torch.cat([gold_tags, stop_tag])

            gold_score[batch_no] = torch.sum(crf_scores[batch_no, token_indices, from_tags[:-1], to_tags[:-1]]) + self.transitions[from_tags[-1], to_tags[-1]]

        return gold_score