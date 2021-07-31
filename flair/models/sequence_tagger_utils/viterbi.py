import torch
import torch.nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence

import flair
from flair.data import Dictionary, Label, List

from .utils import log_sum_exp, START_TAG, STOP_TAG

class ViterbiLoss(torch.nn.Module):
    """
    Viterbi Loss according to sgrvinod (https://github.com/sgrvinod).
    Calculates the loss for each sequence up to its length t.
    """

    def __init__(self, tag_dictionary: Dictionary):
        """
        :param tag_dictionary: tag_dictionary of task
        """
        super(ViterbiLoss, self).__init__()
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def forward(self, features: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of Viterbi Loss

        :param features: CRF scores from CRF forward method in shape (batch size, seq len, tagset size, tagset size)
        :param targets: true tags for sentences which come in as matrix indices.
            CRF scores contain per sentence, per token a (tagset_size x tagset_size) matrix, containing emission score for
            token j + transition prob from previous token i. Means, if we think of our rows as "from tag" and our columns
            as "to tag", the matrix in cell [10,5] would contain the emission score for tag 5 + transition score
            from previous tag 10 and could directly be addressed through the 1-dim indices (10 * tagset_size + 5) = 125,
            if our tagset consists of 12 tags.
        :param lengths: lengths tuple containing sorted lengths and indices from unsorted list
        :return: average Viterbi Loss over batch size
        """
        batch_size = features.size(0)
        seq_len = features.size(1)
        targets = targets.unsqueeze(2)

        # Squeeze crf scores matrices in 1-dim shape and gather scores at targets by matrix indices
        scores_at_targets = torch.gather(features.view(batch_size, seq_len, -1), 2, targets).squeeze(2)
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)[0]
        gold_score = scores_at_targets.sum()

        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(flair.device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # since batch is ordered, we can save computation time by reducing our effective batch_size
            if t == 0:
                # Initially, get scores from <start> tag to all other tags
                scores_upto_t[:batch_size_t] = features[:batch_size_t, t, self.start_tag, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                scores_upto_t[:batch_size_t] = log_sum_exp(features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2), dim=1)

        all_paths_scores = scores_upto_t[:, self.stop_tag].sum()

        viterbi_loss = all_paths_scores - gold_score

        return viterbi_loss


class ViterbiDecoder:
    """
    Viterbi Decoder according to sgrvinod (https://github.com/sgrvinod).
    Decodes a given sequence using the Viterbi algorithm.
    """

    def __init__(self, tag_dictionary: Dictionary):
        """
        :param tag_dictionary: Dictionary of tags for sequence labeling task
        """
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def decode(self, features: torch.Tensor, lengths: torch.Tensor) -> List:
        """
        Decoding function returning the most likely sequence of tags.
        :param features: CRF scores from CRF forward method in shape (batch size, seq len, tagset size, tagset size)
        :param lengths: lengths tuple containing sorted lengths and indices from unsorted list
        :return: decoded sequences
        """
        tags = []
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, seq_len, self.tagset_size).to(flair.device)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, seq_len, self.tagset_size), dtype=torch.long, device=flair.device) * self.stop_tag

        for t in range(seq_len):
            batch_size_t = sum([l > t for l in lengths.values])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t, t] = features[:batch_size_t, t, self.start_tag, :]
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size), dtype=torch.long, device=flair.device) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t, t], backpointers[:batch_size_t, t, :] = torch.max(
                    features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t, t-1].unsqueeze(2),
                    dim=1)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long, device=flair.device)
        pointer = torch.ones((batch_size, 1), dtype=torch.long, device=flair.device) * self.stop_tag

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long, device=flair.device) * self.start_tag)

        # Max + Softmax to get confidence score for predicted label and append label to each token
        confidences = torch.max(softmax(scores_upto_t, dim=2), dim=2)

        for tag_seq, tag_seq_conf, length_seq in zip(decoded, confidences.values, lengths.values):
            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf.item())
                    for tag, conf in list(zip(tag_seq, tag_seq_conf))[1:length_seq]
                ]
            )

        return tags