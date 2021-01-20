import torch
from torch import nn

import flair
from flair.data import Dictionary
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG

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

        crf_scores = emission_scores + self.transitions.unsqueeze(0).unsqueeze(0) # (32, 40, 12, 12)
        return crf_scores