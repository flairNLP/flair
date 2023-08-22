from typing import List, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F

import flair
from flair.data import Sentence, Dictionary, DT, Span, Union, Optional
from flair.embeddings import TokenEmbeddings, DocumentEmbeddings
from flair.training_utils import store_embeddings

log = logging.getLogger("flair")


def contrastive_loss(
    scores: torch.FloatTensor,
    positions: Union[List[int], Tuple[List[int], List[int]]],
    mask: torch.FloatTensor = None,
) -> torch.Tensor:
    batch_size, seq_length = scores.size(0), scores.size(1)
    if len(scores.shape) == 3:
        scores = scores.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        log_probs = masked_log_softmax(scores, mask)
        log_probs = log_probs.view(batch_size, seq_length, seq_length)
        start_positions, end_positions = positions
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, start_positions, end_positions]
    else:
        log_probs = masked_log_softmax(scores, mask)
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, positions]
    return - log_probs.mean()


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

class BinderModel(flair.nn.Classifier[Sentence]):
    """This model implements the BINDER architecture for token classification using contrastive learning and a bi-encoder.
    Paper: https://openreview.net/forum?id=9EAQVEINuum
    """

    def __init__(
        self,
        token_encoder: TokenEmbeddings,
        label_encoder: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        dropout: float = 0.1,
        linear_size: int = 128,
        use_span_width_embeddings: bool = True,
        max_span_width: int = 8,
        init_temperature: float = 0.07,
        start_loss_weight: float = 0.2,
        end_loss_weight: float = 0.2,
        span_loss_weight: float = 0.6,
        threshold_loss_weight: float = 0.5,
        ner_loss_weight: float = 0.5,
    ):
        super().__init__()
        if not token_encoder.subtoken_pooling == "first_last":
            raise RuntimeError("The token encoder must use first_last subtoken pooling when using BINDER model.")

        if not token_encoder.document_embedding == True:
            raise RuntimeError("The token encoder must include the CLS token when using BINDER model.")

        self.token_encoder = token_encoder
        self.label_encoder = label_encoder
        self.label_dictionary = label_dictionary
        self._label_type = label_type

        self.dropout = torch.nn.Dropout(dropout)
        self.label_start_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_end_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_span_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)

        token_embedding_size = self.token_encoder.embedding_length // 2
        self.token_start_linear = torch.nn.Linear(token_embedding_size, linear_size)
        self.token_end_linear = torch.nn.Linear(token_embedding_size, linear_size)

        self.max_span_width = max_span_width
        if use_span_width_embeddings:
            self.token_span_linear = torch.nn.Linear(
                self.token_encoder.embedding_length + linear_size, linear_size
            )
            assert (self.token_encoder.model.config.max_position_embeddings ==
                    self.label_encoder.model.config.max_position_embeddings), \
                "The maximum position embeddings for the token encoder and label encoder must be the same when using " \
                "span width embeddings."
            span_width = self.token_encoder.model.config.max_position_embeddings
            self.width_embeddings = torch.nn.Embedding(span_width, linear_size, padding_idx=0)
        else:
            self.token_span_linear = torch.nn.Linear(self.token_encoder.embedding_length, linear_size)
            self.width_embeddings = None

        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))

        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight

        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Forwards the BINDER model and returns the combined loss."""
        # Quality checks
        if len(data_points) == 0 or not [spans for sentence in data_points for spans in sentence.get_spans()]:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        data_points = [data_points] if not isinstance(data_points, list) else data_points

        # Encode data points
        start_scores, end_scores, span_scores, lengths = self._encode_data_points(data_points)

        assert start_scores.shape == end_scores.shape, "Start and end scores must have the same shape."
        assert span_scores.shape[2] == span_scores.shape[3], "Span scores must be square."

        # Get spans
        start_mask, end_mask, span_mask = self._get_masks(
            start_scores.size(),
            lengths
        )

        # Calculate loss
        total_loss, num_spans = self._calculate_loss(
            data_points, start_scores, end_scores, span_scores, start_mask, end_mask, span_mask
        )

        return total_loss, num_spans

    def _get_token_hidden_states(self, sentences: List[Sentence]) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Returns the token hidden states for the given sentences."""
        names = self.token_encoder.get_names()
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.token_encoder.embedding_length * longest_token_sequence_in_batch,
            device=flair.device,
        )
        all_embs = []
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.token_encoder.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.token_encoder.embedding_length,
            ]
        )
        return torch.LongTensor(lengths), sentence_tensor

    def _encode_data_points(self, sentences) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get hidden states for tokens in shape batch_size x seq_length x hidden_size
        self.token_encoder.embed(sentences)
        lengths, token_hidden_states = self._get_token_hidden_states(sentences)

        # Get hidden states for labels in shape num_types x hidden_size
        labels = [Sentence(label) for label in self.label_dictionary.get_items()]
        self.label_encoder.embed(labels)
        label_hidden_states = torch.stack([label.get_embedding() for label in labels])

        # Extract shapes
        batch_size, org_seq_length, _ = token_hidden_states.size()
        seq_length = org_seq_length * 2 + 1
        num_types, _ = label_hidden_states.size()
        hidden_size = self.token_encoder.embedding_length // 2

        token_hidden_states = token_hidden_states.view(batch_size, org_seq_length, 2, hidden_size).view(batch_size, 2 * org_seq_length, hidden_size)
        cls_hidden_state = torch.stack([sentence.get_embedding() for sentence in sentences])
        # Shape: batch_size x seq_length * 2 (start + end hidden state per token) + 1 (CLS token embedding) x
        # hidden_size // 2 (split the concatenated hidden state in half)
        token_hidden_states = torch.cat([cls_hidden_state.unsqueeze(1), token_hidden_states], dim=1)

        # Reproject + dropout + normalize label hidden states - final shape: num_types x hidden_size
        label_start_output = F.normalize(self.dropout(self.label_start_linear(label_hidden_states)), dim=-1)
        label_end_output = F.normalize(self.dropout(self.label_end_linear(label_hidden_states)), dim=-1)
        # Reproject + dropout + normalize label hidden states - final shape: batch_size x seq_length x hidden_size
        token_start_output = F.normalize(self.dropout(self.token_start_linear(token_hidden_states)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.token_end_linear(token_hidden_states)), dim=-1)

        # obtain start scores for threshold loss - shape: batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * label_start_output.unsqueeze(0) @ token_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * label_end_output.unsqueeze(0) @ token_end_output.transpose(1, 2)

        # get span outputs by concat - shape: batch_size x seq_length x seq_length x hidden_size*2
        token_span_output = torch.cat(
            [
                token_hidden_states.unsqueeze(2).expand(-1, -1, seq_length, -1),
                token_hidden_states.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # If using width_embeddings, add them to the span_output
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=token_hidden_states.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            token_span_output = torch.cat([
                token_span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # Reproject span outputs - shape: batch_size x seq_length x seq_length x hidden_size
        token_span_linear_output = F.normalize(
            self.dropout(self.token_span_linear(token_span_output)).view(batch_size, seq_length * seq_length, -1),
            dim=-1
        )

        # Reproject label embeddings with span linear - shape: num_types x hidden_size
        label_span_linear_output = F.normalize(self.dropout(self.label_span_linear(label_hidden_states)), dim=-1)

        # Obtain the span scores - shape: batch_size x num_types x seq_length x seq_length
        span_scores = self.span_logit_scale.exp() * label_span_linear_output.unsqueeze(0) @ token_span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        return start_scores, end_scores, span_scores, lengths

    def _get_masks(self, scores_size, lengths):
        batch_size, num_types, seq_length = scores_size
        lengths = lengths * 2 + 1

        start_mask = torch.zeros(batch_size, num_types, seq_length, dtype=torch.bool, device=flair.device)
        end_mask = torch.zeros(batch_size, num_types, seq_length, dtype=torch.bool, device=flair.device)
        for i in range(seq_length):
            start_mask[:, :, i] = i % 2
            end_mask[:, :, i] = (i + 1) % 2

        # Include CLS token
        start_mask[:, :, 0] = 1
        end_mask[:, :, 0] = 1

        for i, length in enumerate(lengths):
            start_mask[i, :, length:] = 0
            end_mask[i, :, length:] = 0

        span_mask = (start_mask.unsqueeze(-1) * end_mask.unsqueeze(-2)).triu()
        span_mask[:, :, 0, 0] = 1

        return start_mask, end_mask, span_mask

    def _calculate_loss(self, sentences, start_scores, end_scores, span_scores, start_mask, end_mask, span_mask):
        batch_size, num_types, seq_length = start_scores.size()

        # Flatten first dimension for scores
        flat_start_scores = start_scores.view(batch_size * num_types, seq_length)
        flat_end_scores = end_scores.view(batch_size * num_types, seq_length)
        flat_span_scores = span_scores.view(batch_size * num_types, seq_length, seq_length)

        # Extract all spans from data points. Sequence IDS indicate the index of the sentence in the batch.
        sequence_ids, spans = zip(*[(idx, span) for idx, sentence in enumerate(sentences) for span in sentence.get_spans(self._label_type)])
        # Since we use start and end embeddings seperately, we need to adjust the index of the start and end token
        span_start_positions = [span.tokens[0].idx * 2 - 1 for span in spans]
        span_end_positions = [span.tokens[-1].idx * 2 for span in spans]
        span_types_str = [span.get_label(self._label_type).value for span in spans]
        span_types_id = [self.label_dictionary.get_idx_for_item(span_type) for span_type in span_types_str]

        # Mask the spans for threshold loss
        start_mask[sequence_ids, span_types_id, span_start_positions] = 0
        end_mask[sequence_ids, span_types_id, span_end_positions] = 0
        span_mask[sequence_ids, span_types_id, span_start_positions, span_end_positions] = 0

        start_mask = start_mask.view(batch_size * num_types, seq_length)
        end_mask = end_mask.view(batch_size * num_types, seq_length)
        span_mask = span_mask.view(batch_size * num_types, seq_length, seq_length)

        start_threshold_loss = contrastive_loss(flat_start_scores, [0], start_mask)
        end_threshold_loss = contrastive_loss(flat_end_scores, [0], end_mask)
        span_threshold_loss = contrastive_loss(flat_span_scores, [0, 0], span_mask)

        threshold_loss = (
                self.start_loss_weight * start_threshold_loss +
                self.end_loss_weight * end_threshold_loss +
                self.span_loss_weight * span_threshold_loss
        )

        start_mask = start_mask.view(batch_size, num_types, seq_length)
        end_mask = end_mask.view(batch_size, num_types, seq_length)
        span_mask = span_mask.view(batch_size, num_types, seq_length, seq_length)

        start_mask[sequence_ids, span_types_id, span_start_positions] = 1
        end_mask[sequence_ids, span_types_id, span_end_positions] = 1
        span_mask[sequence_ids, span_types_id, span_start_positions, span_end_positions] = 1

        start_loss = contrastive_loss(start_scores[sequence_ids, span_types_id], span_start_positions, start_mask[sequence_ids, span_types_id])
        end_loss = contrastive_loss(end_scores[sequence_ids, span_types_id], span_end_positions, end_mask[sequence_ids, span_types_id])
        span_loss = contrastive_loss(
            span_scores[sequence_ids, span_types_id],
            (span_start_positions, span_end_positions),
            span_mask[sequence_ids, span_types_id]
        )

        total_loss = (
                self.start_loss_weight * start_loss +
                self.end_loss_weight * end_loss +
                self.span_loss_weight * span_loss
        )

        total_loss = self.ner_loss_weight * total_loss + self.threshold_loss_weight * threshold_loss

        return total_loss, len(spans)

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        with torch.no_grad():
            start_scores, end_scores, span_scores, lengths = self._encode_data_points(sentences)

            start_mask, end_mask, span_mask = self._get_masks(
                start_scores.size(),
                lengths
            )

            span_preds = torch.triu(span_scores > span_scores[:, :, 0:1, 0:1])
            # Perform the element-wise logical operations
            logical_and = start_mask.unsqueeze(3) & end_mask.unsqueeze(2) & span_preds

            # Find the non-zero indices
            sequence_ids, preds, start_indexes, end_indexes = torch.nonzero(logical_and, as_tuple=True)

            # if anything could possibly be predicted
            if len(sentences) > 0:
                # remove previously predicted labels of this type
                for idx, sentence in enumerate(sentences):
                    sentence.remove_labels(label_name)

                    spans_for_sentence = sequence_ids == idx

                    if not spans_for_sentence.any():
                        continue

                    # get the spans
                    predictions = []
                    tokens = sentence.tokens

                    # we revert back start and end predictions to token indices and to ignore CLS token in hidden states
                    start_indexes_for_flair_sentence = start_indexes[spans_for_sentence] // 2
                    end_indexes_for_flair_sentence = end_indexes[spans_for_sentence] // 2
                    start_indexes_for_scores = start_indexes[spans_for_sentence]
                    end_indexes_for_scores = end_indexes[spans_for_sentence]
                    preds_for_sentence = preds[spans_for_sentence]

                    for sentence_start_index, sentence_end_index, score_start_index, score_end_index, pred in zip(
                            start_indexes_for_flair_sentence, end_indexes_for_flair_sentence, start_indexes_for_scores, end_indexes_for_scores, preds_for_sentence
                    ):
                        predictions.append({
                            "span": Span(tokens[sentence_start_index:sentence_end_index]),
                            "start": sentence_start_index,
                            "end": sentence_end_index,
                            "confidence": torch.nn.functional.softmax(span_scores[idx, :, score_start_index, score_end_index], dim=0)[pred].item(),
                            "type": self.label_dictionary.get_item_for_index(pred),
                        })

                    # remove overlaps
                    predictions = self._remove_overlaps(predictions)

                    for prediction in predictions:
                        span = prediction["span"]
                        span.add_label(
                            typename=label_name,
                            value=prediction["type"],
                            score=prediction["confidence"],
                        )

            store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return self._calculate_loss(
                sentences, start_scores, end_scores, span_scores, start_mask, end_mask, span_mask
            )

        return None

    @staticmethod
    def _remove_overlaps(predictions):
        # Sort the predictions based on the start values
        sorted_predictions = sorted(predictions, key=lambda x: x['start'])

        # Initialize a list to store non-overlapping predictions
        non_overlapping = []

        for prediction in sorted_predictions:
            if not non_overlapping:
                non_overlapping.append(prediction)
            else:
                # Check for overlap with the last prediction in the non-overlapping list
                last_prediction = non_overlapping[-1]
                if prediction['start'] > last_prediction['end']:
                    non_overlapping.append(prediction)
                else:
                    # Handle the overlap, e.g., by choosing the one with higher confidence
                    if prediction['confidence'] > last_prediction['confidence']:
                        non_overlapping[-1] = prediction  # Replace the last prediction

        return non_overlapping
