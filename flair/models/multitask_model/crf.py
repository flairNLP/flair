import torch
from torch import nn

class CRF(nn.Module):
    """
    Conditional Random Field
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: hidden size of RNN output
        :param tagset_size: number of tag from tag dictionary
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, tagset_size)
        self.transitions = torch.nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.data.zero_()

    def forward(self, rnn_features):
        """
        Forward propagation.
        :param rnn_features: output from RNN unit in shape (batch size, seq len, hidden size)
        :return: CRF scores (emission scores for each token + transitions to every state) in
        shape (batch_size, seq len, tagset size, tagset size)
        """
        batch_size = rnn_features.size(0)
        seq_len = rnn_features.size(1)

        emission_scores = self.emission(rnn_features)
        emission_scores = emission_scores.unsqueeze(2).expand(batch_size, seq_len, self.tagset_size, self.tagset_size)

        crf_scores = emission_scores + self.transitions.unsqueeze(0).unsqueeze(0)
        return crf_scores