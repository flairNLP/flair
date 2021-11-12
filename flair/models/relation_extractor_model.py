import logging
from pathlib import Path
from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn

import flair.embeddings
import flair.nn
from flair.data import DataPoint, RelationLabel, Span, Sentence
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class RelationExtractor(flair.nn.DefaultClassifier):

    def __init__(
            self,
            embeddings: Union[flair.embeddings.TokenEmbeddings],
            label_type: str,
            entity_label_type: str,
            train_on_gold_pairs_only: bool = False,
            entity_pair_filters: List[Tuple[str, str]] = None,
            pooling_operation: str = "first_last",
            dropout_value: float = 0.0,
            locked_dropout_value: float = 0.1,
            word_dropout_value: float = 0.0,
            non_linear_decoder: Optional[int] = 2048,
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

        # set embeddings
        self.embeddings: flair.embeddings.TokenEmbeddings = embeddings

        # set relation and entity label types
        self._label_type = label_type
        self.entity_label_type = entity_label_type

        # whether to use gold entity pairs, and whether to filter entity pairs by type
        self.train_on_gold_pairs_only = train_on_gold_pairs_only
        if entity_pair_filters is not None:
            self.entity_pair_filters = set(entity_pair_filters)
        else:
            self.entity_pair_filters = None

        # init dropouts
        self.dropout_value = dropout_value
        self.dropout = torch.nn.Dropout(dropout_value)
        self.locked_dropout_value = locked_dropout_value
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout_value)
        self.word_dropout_value = word_dropout_value
        self.word_dropout = flair.nn.WordDropout(word_dropout_value)

        # pooling operation to get embeddings for entites
        self.pooling_operation = pooling_operation
        relation_representation_length = 2 * embeddings.embedding_length
        if self.pooling_operation == 'first_last':
            relation_representation_length *= 2
        if type(self.embeddings) == flair.embeddings.TransformerDocumentEmbeddings:
            relation_representation_length = embeddings.embedding_length

        # entity pairs could also be no relation at all, add default value for this case to dictionary
        self.label_dictionary.add_item('O')

        # decoder can be linear or nonlinear
        self.non_linear_decoder = non_linear_decoder
        if self.non_linear_decoder:
            self.decoder_1 = nn.Linear(relation_representation_length, non_linear_decoder)
            self.nonlinearity = torch.nn.ReLU()
            self.decoder_2 = nn.Linear(non_linear_decoder, len(self.label_dictionary))
            nn.init.xavier_uniform_(self.decoder_1.weight)
            nn.init.xavier_uniform_(self.decoder_2.weight)
        else:
            self.decoder = nn.Linear(relation_representation_length, len(self.label_dictionary))
            nn.init.xavier_uniform_(self.decoder.weight)

        self.to(flair.device)

    def add_entity_markers(self, sentence, span_1, span_2):

        text = ""

        entity_one_is_first = None
        offset = 0
        for token in sentence:
            if token == span_2[0]:
                if entity_one_is_first is None: entity_one_is_first = False
                offset += 1
                text += " <e2>"
                span_2_startid = offset
            if token == span_1[0]:
                offset += 1
                text += " <e1>"
                if entity_one_is_first is None: entity_one_is_first = True
                span_1_startid = offset

            text += " " + token.text

            if token == span_1[-1]:
                offset += 1
                text += " </e1>"
                span_1_stopid = offset
            if token == span_2[-1]:
                offset += 1
                text += " </e2>"
                span_2_stopid = offset

            offset += 1

        expanded_sentence = Sentence(text, use_tokenizer=False)

        expanded_span_1 = Span([expanded_sentence[span_1_startid - 1]])
        expanded_span_2 = Span([expanded_sentence[span_2_startid - 1]])

        return expanded_sentence, (expanded_span_1, expanded_span_2) \
            if entity_one_is_first else (expanded_span_2, expanded_span_1)

    def forward_pass(self,
                     sentences: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ):

        empty_label_candidates = []
        entity_pairs = []
        labels = []
        sentences_to_label = []

        for sentence in sentences:

            # super lame: make dictionary to find relation annotations for a given entity pair
            relation_dict = {}
            for relation_label in sentence.get_labels(self.label_type):
                relation_label: RelationLabel = relation_label
                relation_dict[create_position_string(relation_label.head, relation_label.tail)] = relation_label

            # get all entity spans
            span_labels = sentence.get_labels(self.entity_label_type)

            # go through cross product of entities, for each pair concat embeddings
            for span_label in span_labels:
                span_1 = span_label.span

                for span_label_2 in span_labels:
                    span_2 = span_label_2.span

                    if span_1 == span_2:
                        continue

                    # filter entity pairs according to their tags if set
                    if (self.entity_pair_filters is not None
                            and (span_label.value, span_label_2.value) not in self.entity_pair_filters):
                        continue

                    position_string = create_position_string(span_1, span_2)

                    # get gold label for this relation (if one exists)
                    if position_string in relation_dict:
                        relation_label: RelationLabel = relation_dict[position_string]
                        label = relation_label.value

                    # if there is no gold label for this entity pair, set to 'O' (no relation)
                    else:
                        if self.train_on_gold_pairs_only: continue  # skip 'O' labels if training on gold pairs only
                        label = 'O'

                    entity_pairs.append((span_1, span_2))

                    labels.append([label])

                    # if predicting, also remember sentences and label candidates
                    if return_label_candidates:
                        candidate_label = RelationLabel(head=span_1, tail=span_2, value=None, score=None)
                        empty_label_candidates.append(candidate_label)
                        sentences_to_label.append(span_1[0].sentence)

        # if there's at least one entity pair in the sentence
        if len(entity_pairs) > 0:

            # embed sentences and get embeddings for each entity pair
            self.embeddings.embed(sentences)
            relation_embeddings = []

            # get embeddings
            for entity_pair in entity_pairs:
                span_1 = entity_pair[0]
                span_2 = entity_pair[1]

                if self.pooling_operation == "first_last":
                    embedding = torch.cat([span_1.tokens[0].get_embedding(),
                                           span_1.tokens[-1].get_embedding(),
                                           span_2.tokens[0].get_embedding(),
                                           span_2.tokens[-1].get_embedding()])
                else:
                    embedding = torch.cat([span_1.tokens[0].get_embedding(), span_2.tokens[0].get_embedding()])

                relation_embeddings.append(embedding)

            # stack and drop out (squeeze and unsqueeze)
            all_relations = torch.stack(relation_embeddings).unsqueeze(1)

            all_relations = self.dropout(all_relations)
            all_relations = self.locked_dropout(all_relations)
            all_relations = self.word_dropout(all_relations)

            all_relations = all_relations.squeeze(1)

            # send through decoder
            if self.non_linear_decoder:
                sentence_relation_scores = self.decoder_2(self.nonlinearity(self.decoder_1(all_relations)))
            else:
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
            "embeddings": self.embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_type": self.entity_label_type,
            "loss_weights": self.loss_weights,
            "pooling_operation": self.pooling_operation,
            "dropout_value": self.dropout_value,
            "locked_dropout_value": self.locked_dropout_value,
            "word_dropout_value": self.word_dropout_value,
            "entity_pair_filters": self.entity_pair_filters,
            "non_linear_decoder": self.non_linear_decoder,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = RelationExtractor(
            embeddings=state["embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            entity_label_type=state["entity_label_type"],
            loss_weights=state["loss_weights"],
            pooling_operation=state["pooling_operation"],
            dropout_value=state["dropout_value"],
            locked_dropout_value=state["locked_dropout_value"],
            word_dropout_value=state["word_dropout_value"],
            entity_pair_filters=state["entity_pair_filters"],
            non_linear_decoder=state["non_linear_decoder"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    @property
    def label_type(self):
        return self._label_type

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["relations-fast"] = "/".join([hu_path, "relations-fast", "relations-fast.pt"])
        model_map["relations"] = "/".join([hu_path, "relations", "relations.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name


def create_position_string(head: Span, tail: Span) -> str:
    return f"{head.id_text} -> {tail.id_text}"
