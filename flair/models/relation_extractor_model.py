import logging
from typing import List, Union, Tuple

import torch
import torch.nn as nn

import flair.embeddings
import flair.nn
from flair.data import DataPoint, RelationLabel, Span, Sentence

log = logging.getLogger("flair")


class RelationExtractor(flair.nn.DefaultClassifier):

    def __init__(
            self,
            token_embeddings: flair.embeddings.TokenEmbeddings,
            label_type: str = None,
            span_label_type: str = None,
            use_entity_markers: bool = False,
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
        self.use_entity_markers = use_entity_markers

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

    def forward_pass_markup(self,
                            sentences: Union[List[DataPoint], DataPoint],
                            return_label_candidates: bool = False,
                            ):

        for sentence in sentences:

            # super lame: make dictionary to find relation annotations for a given entity pair
            relation_dict = {}
            for relation_label in sentence.get_labels(self.label_type):
                relation_label: RelationLabel = relation_label
                relation_dict[create_position_string(relation_label.head, relation_label.tail)] = relation_label

            # get all entity spans
            span_labels = sentence.get_labels(self.span_label_type)

            # go through cross product of entities, for each pair concat embeddings
            for span_label in span_labels:
                span_1 = span_label.span

                for span_label_2 in span_labels:
                    span_2 = span_label_2.span

                    if span_1 == span_2:
                        continue

                    if (self.use_entity_pairs is not None
                            and (span_label.value, span_label_2.value) not in self.use_entity_pairs):
                        continue

                    text = ""

                    entity_one_is_first = None
                    for token in sentence:
                        if token == span_1[0]:
                            text += " <e1>"
                            if entity_one_is_first is None: entity_one_is_first = True
                        if token == span_2[0]:
                            if entity_one_is_first is None: entity_one_is_first = False
                            text += " <e2>"

                        text += " " + token.text

                        if token == span_1[-1]:
                            text += " </e1>"
                        if token == span_2[-1]:
                            text += " </e2>"

                    expanded_sentence = Sentence(text, use_tokenizer=False)
                    expanded_sentence.original = sentence

                    for token in span_1:
                        offset = 0 if entity_one_is_first else 2
                        expanded_sentence[token.idx + offset] \
                            .set_label(self.span_label_type, token.get_tag(self.span_label_type).value)

                    for token in span_2:
                        offset = 2 if entity_one_is_first else 0
                        expanded_sentence[token.idx + offset] \
                            .set_label(self.span_label_type, token.get_tag(self.span_label_type).value)

                    spans = expanded_sentence.get_spans(self.span_label_type)

                    position_string = create_position_string(span_1, span_2)

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

                    expanded_sentence.add_complex_label(self.label_type,
                                                        RelationLabel(head=spans[0], tail=spans[1], value=label))

                    print(expanded_sentence)

    def add_entity_markers(self, sentence, span_1, span_2):
        # print()
        # print()
        # print(sentence)
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

        if expanded_span_1.text != '<e1>': asd
        if expanded_span_2.text != '<e2>': asd

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

        sentences_to_embed = [] if self.use_entity_markers else sentences

        for sentence in sentences:

            # super lame: make dictionary to find relation annotations for a given entity pair
            relation_dict = {}
            for relation_label in sentence.get_labels(self.label_type):
                relation_label: RelationLabel = relation_label
                relation_dict[create_position_string(relation_label.head, relation_label.tail)] = relation_label

            # get all entity spans
            span_labels = sentence.get_labels(self.span_label_type)

            # go through cross product of entities, for each pair concat embeddings
            for span_label in span_labels:
                span_1 = span_label.span

                for span_label_2 in span_labels:
                    span_2 = span_label_2.span

                    if span_1 == span_2:
                        continue

                    if (self.use_entity_pairs is not None
                            and (span_label.value, span_label_2.value) not in self.use_entity_pairs):
                        continue

                    position_string = create_position_string(span_1, span_2)

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

                    if self.use_entity_markers:
                        expanded_sentence, expanded_entities = self.add_entity_markers(sentence, span_1, span_2)
                        sentences_to_embed.append(expanded_sentence)
                        entity_pairs.append(expanded_entities)
                    else:
                        entity_pairs.append((span_1, span_2))

                    labels.append([label])

                    # if predicting, also remember sentences and label candidates
                    if return_label_candidates:
                        candidate_label = RelationLabel(head=span_1, tail=span_2, value=None, score=None)
                        empty_label_candidates.append(candidate_label)
                        sentences_to_label.append(span_1[0].sentence)

        if len(labels) > 0:

            max_relations_in_batch = len(sentences) * 2
            if len(sentences_to_embed) > max_relations_in_batch:
                sentence_embed_steps = [sentences_to_embed[x: x + max_relations_in_batch]
                                        for x in range(0, len(sentences_to_embed), max_relations_in_batch)]
                entity_pairs_steps = [entity_pairs[x: x + max_relations_in_batch]
                                        for x in range(0, len(entity_pairs), max_relations_in_batch)]
            else:
                sentence_embed_steps = [sentences_to_embed]
                entity_pairs_steps = [entity_pairs]

            relation_embeddings = []
            detach = False
            for sentences_to_embed, entity_pairs in zip(sentence_embed_steps, entity_pairs_steps):

                # embed sentences
                self.token_embeddings.embed(sentences_to_embed)

                # get embeddings
                for entity_pair in entity_pairs:
                    span_1 = entity_pair[0]
                    span_2 = entity_pair[1]
                    embedding = torch.cat([span_1.tokens[0].get_embedding(), span_2.tokens[0].get_embedding()])
                    if detach:
                        embedding = embedding.detach()
                    #     print("detach")
                    # print(embedding)
                    relation_embeddings.append(embedding)

                detach = True

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
