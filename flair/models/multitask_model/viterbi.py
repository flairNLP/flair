import torch
import torch.nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import flair
from flair.data import Dictionary, Label
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG, log_sum_exp_refactor

class ViterbiLoss(torch.nn.Module):
    """
    Viterbi Loss.
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

    def forward(self, features: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
        """
        Forward propagation.

        :param features: CRF scores from forward method
        :param tags: true tags for sentences
        :param lengths_sorted_indices: sorted lengths indices
        :return: Viterbi Loss as average from (forward score - gold score)
        """
        batch_size = features.size(0)
        seq_len = features.size(1)
        targets = targets.unsqueeze(2)

        scores_at_targets = torch.gather(features.view(batch_size, seq_len, -1), 2, targets).squeeze(2)
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)[0]
        gold_score = scores_at_targets.sum()

        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(flair.device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = features[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp_refactor(features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2), dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.stop_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss


class ViterbiDecoder:
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_dictionary: Dictionary):
        """
        :param tag_map: tag map
        """
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def decode(self, features, lengths):
        """
        :param features: CRF scores
        :param lengths: word sequence lengths
        :return: decoded sequences
        """
        tags = []
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, seq_len, self.tagset_size)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, seq_len, self.tagset_size), dtype=torch.long) * self.stop_tag

        for t in range(seq_len):
            batch_size_t = sum([l > t for l in lengths.values])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t, t] = features[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t, t], backpointers[:batch_size_t, t, :] = torch.max(
                    features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t, t-1].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long)
        pointer = torch.ones((batch_size, 1), dtype=torch.long) * self.stop_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long) * self.start_tag)

        # Max + Softmax to get confidence score for predicted label
        confidences = torch.max(softmax(scores_upto_t, dim=2), dim=2)


        for tag_seq, tag_seq_conf, length_seq in zip(decoded, confidences.values, lengths.values):
            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf.item())
                    for tag, conf in list(zip(tag_seq, tag_seq_conf))[1:length_seq] # this slices only for the relevant part of the seq
                ]
            )

        return tags