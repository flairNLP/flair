import logging
from typing import List, Union, Dict, Optional, Tuple

import torch
import torch.nn as nn

import flair.embeddings
import flair.nn
from flair.data import DataPoint, RelationLabel, Span

log = logging.getLogger("flair")


class RelationExtractor(flair.nn.DefaultClassifier):

    def __init__(
            self,
            token_embeddings: flair.embeddings.TokenEmbeddings,
            label_type: str = None,
            span_label_type: str = None,
            use_gold_spans: bool = False,
            use_entity_pairs: List[Tuple[str, str]] = None,
            pooling_operation: str = "first_last",
            dropout_value: float = 0.0,
            **classifierargs,
    ):
        """
        Initializes a RelationClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """
        super(RelationExtractor, self).__init__(**classifierargs)

        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings
        self._label_type = label_type
        self.span_label_type = span_label_type

        self.use_gold_spans = use_gold_spans
        self.pooling_operation = pooling_operation

        self.dropout_value = dropout_value
        self.dropout = torch.nn.Dropout(dropout_value)

        if use_entity_pairs is not None:
            self.use_entity_pairs = set(use_entity_pairs)
        else:
            self.use_entity_pairs = None

        relation_representation_length = 2 * token_embeddings.embedding_length
        if self.pooling_operation == 'first_last':
            relation_representation_length *= 2

        # entity pairs could also be no relation at all, add default value for this case to dictionary
        self.label_dictionary.add_item('O')

        self.decoder = nn.Linear(relation_representation_length, len(self.label_dictionary))

        nn.init.xavier_uniform_(self.decoder.weight)

        self.to(flair.device)

    def forward_pass(self,
                     sentences: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ):

        self.token_embeddings.embed(sentences)

        # entity_pairs = []
        empty_label_candidates = []
        relation_embeddings = []
        labels = []
        sentences_to_label = []

        for sentence in sentences:

            # super lame: make dictionary to find relation annotations for a given entity pair
            relation_dict = {}
            for relation_label in sentence.get_labels(self.label_type):
                relation_label: RelationLabel = relation_label
                relation_dict[create_position_string(relation_label.head, relation_label.tail)] = relation_label

            # get all entity spans
            span_labels = sentence.get_labels(self.span_label_type)

            # get embedding for each entity
            span_embeddings = []
            for span_label in span_labels:
                span: Span = span_label.span
                if self.pooling_operation == "first":
                    span_embeddings.append(span.tokens[0].get_embedding())
                if self.pooling_operation == "first_last":
                    span_embeddings.append(torch.cat([span.tokens[0].get_embedding(), span.tokens[-1].get_embedding()]))

            # go through cross product of entities, for each pair concat embeddings
            for span_label, embedding in zip(span_labels, span_embeddings):
                span = span_label.span

                for span_label_2, embedding_2 in zip(span_labels, span_embeddings):
                    span_2 = span_label_2.span

                    if span == span_2:
                        continue

                    if (self.use_entity_pairs is not None
                        and (span_label.value, span_label_2.value) not in self.use_entity_pairs):
                        continue

                    position_string = create_position_string(span, span_2)

                    # get gold label for this relation (if one exists)
                    if position_string in relation_dict:
                        relation_label: RelationLabel = relation_dict[position_string]
                        label = relation_label.value
                    # if using gold spans only, skip all entity pairs that are not in gold data
                    elif self.use_gold_spans:
                        continue
                    else:
                        # if no gold label exists, and all spans are used, label defaults to 'O' (no relation)
                        label = 'O'

                    labels.append([label])

                    relation_embeddings.append(torch.cat([embedding, embedding_2]))

                    # if predicting, also remember sentences and label candidates
                    if return_label_candidates:
                        candidate_label = RelationLabel(head=span, tail=span_2, value=None, score=None)
                        empty_label_candidates.append(candidate_label)
                        sentences_to_label.append(span[0].sentence)

        if len(labels) > 0:

            all_relations = torch.stack(relation_embeddings)

            all_relations = self.dropout(all_relations)

            sentence_relation_scores = self.decoder(all_relations)

        else:
            sentence_relation_scores = None

        # return either scores and gold labels (for loss calculation), or include label candidates for prediction
        result_tuple = (sentence_relation_scores, labels)

        if return_label_candidates:
            result_tuple += (sentences_to_label, empty_label_candidates)

        return result_tuple

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "token_embeddings": self.token_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "span_label_type": self.span_label_type,
            "loss_weights": self.loss_weights,
            "pooling_operation": self.pooling_operation,
            "dropout_value": self.dropout_value,
            "use_entity_pairs": self.use_entity_pairs,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = RelationExtractor(
            token_embeddings=state["token_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            span_label_type=state["span_label_type"],
            loss_weights=state["loss_weights"],
            pooling_operation=state["pooling_operation"],
            dropout_value=state["dropout_value"],
            use_entity_pairs=state["use_entity_pairs"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    @property
    def label_type(self):
        return self._label_type


def create_position_string(head: Span, tail: Span) -> str:
    return f"{head.id_text} -> {tail.id_text}"
