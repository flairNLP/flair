import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import flair
from flair.data import Dictionary
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
    """
    def forward(self, features: torch.Tensor, tags: torch.Tensor, lengths_sorted_indices: torch.Tensor):
        batch_size = features.size(0)
        seq_len = features.size(1)

        scores_up_to_t = torch.zeros(
            batch_size,
            seq_len + 1,
            self.tagset_size,
            dtype=torch.float,
            device=flair.device,
        )

        start_mask = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        start_mask[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
        scores_up_to_t[:, 0, :] = start_mask.unsqueeze(0).repeat(batch_size, 1)

        for token_num in range(max(lengths)):
            crf_score = features[:, token_num, :]
            score_up_to_t = scores_up_to_t[:, token_num, :]

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
            gold_tags = tags[batch_no, : lengths[batch_no]] # gold_tags = [2,5,1,2,4,5,2]
            from_tags = torch.cat([start_tag, gold_tags]) # von 10 zu allen anderen
            to_tags = torch.cat([gold_tags, stop_tag]) # von allen anderen zu 11

            gold_score[batch_no] = torch.sum(crf_scores[batch_no, token_indices, from_tags[:-1], to_tags[:-1]]) + self.transitions[from_tags[-1], to_tags[-1]]

        return gold_score
    """

class ViterbiDecoder:
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def decode(self, scores, lengths):
        """
        :param scores: CRF scores
        :param lengths: word sequence lengths
        :return: decoded sequences
        """
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.tagset_size), dtype=torch.long) * self.end_tag

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long)
        pointer = torch.ones((batch_size, 1),
                             dtype=torch.long) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long) * self.start_tag)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long) * self.start_tag],
                            dim=1)

        return decoded