from typing import Tuple

import numpy as np
import torch
import torch.nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence

import flair
from flair.data import Dictionary, Label, List, Sentence

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


class ViterbiLoss(torch.nn.Module):
    """Calculates the loss for each sequence up to its length t."""

    def __init__(self, tag_dictionary: Dictionary) -> None:
        """Create an instance of the Viterbi loss.

        Args:
            tag_dictionary: tag_dictionary of task
        """
        super().__init__()
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def forward(self, features_tuple: tuple, targets: torch.Tensor) -> torch.Tensor:
        """Forward propagation of Viterbi Loss.

        Args:
            features_tuple: CRF scores from forward method in shape (batch size, seq len, tagset size, tagset size), lengths of sentences in batch, transitions from CRF
            targets: true tags for sentences which will be converted to matrix indices.

        Returns: summed Viterbi Loss over all data points
        """
        features, lengths, transitions = features_tuple

        batch_size = features.size(0)
        seq_len = features.size(1)

        targets, targets_matrix_indices = self._format_targets(targets, lengths)
        targets_matrix_indices = torch.tensor(targets_matrix_indices, dtype=torch.long).unsqueeze(2).to(flair.device)

        # scores_at_targets[range(features.shape[0]), lengths.values -1]
        # Squeeze crf scores matrices in 1-dim shape and gather scores at targets by matrix indices
        scores_at_targets = torch.gather(features.view(batch_size, seq_len, -1), 2, targets_matrix_indices)
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)[0]
        transitions_to_stop = transitions[
            np.repeat(self.stop_tag, features.shape[0]),
            [target[length - 1] for target, length in zip(targets, lengths)],
        ]
        gold_score = scores_at_targets.sum() + transitions_to_stop.sum()

        scores_upto_t = torch.zeros(batch_size, self.tagset_size, device=flair.device, dtype=features.dtype)

        for t in range(max(lengths)):
            batch_size_t = sum(
                [length > t for length in lengths]
            )  # since batch is ordered, we can save computation time by reducing our effective batch_size

            if t == 0:
                # Initially, get scores from <start> tag to all other tags
                scores_upto_t[:batch_size_t] = (
                    scores_upto_t[:batch_size_t] + features[:batch_size_t, t, :, self.start_tag]
                )
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                scores_upto_t[:batch_size_t] = self._log_sum_exp(
                    features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(1), dim=2
                )

        all_paths_scores = self._log_sum_exp(scores_upto_t + transitions[self.stop_tag].unsqueeze(0), dim=1).sum()

        viterbi_loss = all_paths_scores - gold_score

        return viterbi_loss

    @staticmethod
    def _log_sum_exp(tensor, dim):
        """Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

        Args:
            tensor: tensor
            dim: dimension to calculate log-sum-exp of

        Returns: log-sum-exp
        """
        m, _ = torch.max(tensor, dim)
        m_expanded = m.unsqueeze(dim).expand_as(tensor)
        return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))

    def _format_targets(self, targets: torch.Tensor, lengths: torch.IntTensor):
        """Formats targets into matrix indices.

        CRF scores contain per sentence, per token a (tagset_size x tagset_size) matrix, containing emission score for
        token j + transition prob from previous token i. Means, if we think of our rows as "to tag" and our columns
        as "from tag", the matrix in cell [10,5] would contain the emission score for tag 10 + transition score
        from previous tag 5 and could directly be addressed through the 1-dim indices (10 + tagset_size * 5) = 70,
        if our tagset consists of 12 tags.

        Args:
            targets: targets as in tag dictionary
            lengths: lengths of sentences in batch
        """
        targets_per_sentence = []

        targets_list = targets.tolist()
        for cut in lengths:
            targets_per_sentence.append(targets_list[:cut])
            targets_list = targets_list[cut:]

        for t in targets_per_sentence:
            t += [self.tag_dictionary.get_idx_for_item(STOP_TAG)] * (int(lengths.max().item()) - len(t))

        matrix_indices = [
            [self.tag_dictionary.get_idx_for_item(START_TAG) + (s[0] * self.tagset_size)]
            + [s[i] + (s[i + 1] * self.tagset_size) for i in range(len(s) - 1)]
            for s in targets_per_sentence
        ]

        return targets_per_sentence, matrix_indices


class ViterbiDecoder:
    """Decodes a given sequence using the Viterbi algorithm."""

    def __init__(self, tag_dictionary: Dictionary) -> None:
        """Initialize the Viterbi Decoder.

        Args:
            tag_dictionary: Dictionary of tags for sequence labeling task
        """
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def decode(
        self, features_tuple: tuple, probabilities_for_all_classes: bool, sentences: List[Sentence]
    ) -> Tuple[List, List]:
        """Decoding function returning the most likely sequence of tags.

        Args:
            features_tuple: CRF scores from forward method in shape (batch size, seq len, tagset size, tagset size), lengths of sentence in batch, transitions of CRF
            probabilities_for_all_classes: whether to return probabilities for all tags
            sentences: list of the respective sentences with extracted features.

        Returns: decoded sequences
        """
        features, lengths, transitions = features_tuple
        all_tags = []

        batch_size = features.size(0)
        seq_len = features.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, seq_len + 1, self.tagset_size, dtype=features.dtype).to(flair.device)
        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = (
            torch.ones((batch_size, seq_len + 1, self.tagset_size), dtype=torch.long, device=flair.device)
            * self.stop_tag
        )

        for t in range(seq_len):
            batch_size_t = sum([length > t for length in lengths])  # effective batch size (sans pads) at this timestep
            terminates = [i for i, length in enumerate(lengths) if length == t + 1]

            if t == 0:
                scores_upto_t[:batch_size_t, t] = features[:batch_size_t, t, :, self.start_tag]
                backpointers[:batch_size_t, t, :] = (
                    torch.ones((batch_size_t, self.tagset_size), dtype=torch.long) * self.start_tag
                )
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t, t], backpointers[:batch_size_t, t, :] = torch.max(
                    features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t, t - 1].unsqueeze(1), dim=2
                )

            # If sentence is over, add transition to STOP-tag
            if terminates:
                scores_upto_t[terminates, t + 1], backpointers[terminates, t + 1, :] = torch.max(
                    scores_upto_t[terminates, t].unsqueeze(1) + transitions[self.stop_tag].unsqueeze(0), dim=2
                )

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long, device=flair.device)
        pointer = torch.ones((batch_size, 1), dtype=torch.long, device=flair.device) * self.stop_tag

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)

        # Sanity check
        assert torch.equal(
            decoded[:, 0], torch.ones((batch_size), dtype=torch.long, device=flair.device) * self.start_tag
        )

        # remove start-tag and backscore to stop-tag
        scores_upto_t = scores_upto_t[:, :-1, :]
        decoded = decoded[:, 1:]

        # Max + Softmax to get confidence score for predicted label and append label to each token
        scores = softmax(scores_upto_t, dim=2)
        confidences = torch.max(scores, dim=2)

        tags = []
        for tag_seq, tag_seq_conf, length_seq in zip(decoded, confidences.values, lengths):
            tags.append(
                [
                    (self.tag_dictionary.get_item_for_index(tag), conf.item())
                    for tag, conf in list(zip(tag_seq, tag_seq_conf))[:length_seq]
                ]
            )

        if probabilities_for_all_classes:
            all_tags = self._all_scores_for_token(scores.cpu(), decoded.cpu(), lengths, sentences)

        return tags, all_tags

    def _all_scores_for_token(
        self,
        score_tensor: torch.Tensor,
        tag_sequences: torch.Tensor,
        lengths: torch.IntTensor,
        sentences: List[Sentence],
    ):
        """Returns all scores for each tag in tag dictionary."""
        scores = score_tensor.numpy()
        for i_batch, (batch, tag_seq) in enumerate(zip(scores, tag_sequences)):
            for i, (tag_id, tag_scores) in enumerate(zip(tag_seq, batch)):
                tag_id_int = tag_id if isinstance(tag_id, int) else int(tag_id.item())

                if tag_id_int != np.argmax(tag_scores):
                    swap_index_score = int(np.argmax(tag_scores))
                    scores[i_batch][i][tag_id_int], scores[i_batch][i][swap_index_score] = (
                        scores[i_batch][i][swap_index_score],
                        scores[i_batch][i][tag_id_int],
                    )
        prob_tags_per_sentence = []
        for scores_sentence, length, sentence in zip(scores, lengths, sentences):
            scores_sentence = scores_sentence[:length]
            prob_tags_per_sentence.append(
                [
                    [
                        Label(token, self.tag_dictionary.get_item_for_index(score_id), score)
                        for score_id, score in enumerate(score_dist)
                    ]
                    for score_dist, token in zip(scores_sentence, sentence)
                ]
            )
        return prob_tags_per_sentence
