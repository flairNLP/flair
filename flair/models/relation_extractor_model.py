import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch

import flair.embeddings
import flair.nn
from flair.data import DataPoint, Relation, Sentence, Span
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class RelationExtractor(flair.nn.DefaultClassifier[Sentence]):
    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
        label_type: str,
        entity_label_type: str,
        entity_pair_filters: List[Tuple[str, str]] = None,
        pooling_operation: str = "first_last",
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

        # pooling operation to get embeddings for entites
        self.pooling_operation = pooling_operation
        relation_representation_length = 2 * embeddings.embedding_length
        if self.pooling_operation == "first_last":
            relation_representation_length *= 2
        self.relation_representation_length = relation_representation_length
        super(RelationExtractor, self).__init__(**classifierargs, final_embedding_size=relation_representation_length)

        # set embeddings
        self.embeddings: flair.embeddings.TokenEmbeddings = embeddings

        # set relation and entity label types
        self._label_type = label_type
        self.entity_label_type = entity_label_type

        # whether to use gold entity pairs, and whether to filter entity pairs by type
        if entity_pair_filters is not None:
            self.entity_pair_filters: Optional[Set[Tuple[str, str]]] = set(entity_pair_filters)
        else:
            self.entity_pair_filters = None

        self.to(flair.device)

    def add_entity_markers(self, sentence, span_1, span_2):

        text = ""

        entity_one_is_first = None
        offset = 0
        for token in sentence:
            if token == span_2[0]:
                if entity_one_is_first is None:
                    entity_one_is_first = False
                offset += 1
                text += " <e2>"
                span_2_startid = offset
            if token == span_1[0]:
                offset += 1
                text += " <e1>"
                if entity_one_is_first is None:
                    entity_one_is_first = True
                span_1_startid = offset

            text += " " + token.text

            if token == span_1[-1]:
                offset += 1
                text += " </e1>"
            if token == span_2[-1]:
                offset += 1
                text += " </e2>"

            offset += 1

        expanded_sentence = Sentence(text, use_tokenizer=False)

        expanded_span_1 = Span([expanded_sentence[span_1_startid - 1]])
        expanded_span_2 = Span([expanded_sentence[span_2_startid - 1]])

        return (
            expanded_sentence,
            (
                expanded_span_1,
                expanded_span_2,
            )
            if entity_one_is_first
            else (expanded_span_2, expanded_span_1),
        )

    def _get_labels(self, sentences: List[Sentence]) -> List[List[str]]:
        labels = []
        for sentence in sentences:

            relation_dict = {}
            for label in sentence.get_labels(self.label_type):
                relation_dict[create_position_string(label.data_point.first, label.data_point.second)] = label.value

            for relation in self._get_valid_relations(sentence):
                position_string = create_position_string(relation.first, relation.second)

                # get gold label for this relation if one exists or O otherwise
                label = relation_dict.get(position_string, "O")
                labels.append([label])
        return labels

    def _get_valid_relations(self, sentence: Sentence) -> List[Relation]:
        entity_pairs = []
        entity_spans = sentence.get_spans(self.entity_label_type)

        for span_1 in entity_spans:
            for span_2 in entity_spans:
                if span_1 == span_2:
                    continue

                # filter entity pairs according to their tags if set
                if (
                    self.entity_pair_filters is not None
                    and (
                        span_1.get_label(self.entity_label_type).value,
                        span_2.get_label(self.entity_label_type).value,
                    )
                    not in self.entity_pair_filters
                ):
                    continue

                entity_pairs.append(Relation(span_1, span_2))
        return entity_pairs

    def _get_prediction_data_points(self, sentences: List[Sentence]) -> List[DataPoint]:
        entity_pairs: List[DataPoint] = []

        for sentence in sentences:
            entity_pairs.extend(self._get_valid_relations(sentence))
        return entity_pairs

    def _prepare_tensors(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, ...]:
        entity_pairs = []

        for sentence in sentences:
            entity_pairs.extend(self._get_valid_relations(sentence))

        self.embeddings.embed(sentences)
        relation_embeddings = []

        # get embeddings
        for entity_pair in entity_pairs:
            span_1 = entity_pair.first
            span_2 = entity_pair.second

            if self.pooling_operation == "first_last":
                embedding = torch.cat(
                    [
                        span_1.tokens[0].get_embedding(),
                        span_1.tokens[-1].get_embedding(),
                        span_2.tokens[0].get_embedding(),
                        span_2.tokens[-1].get_embedding(),
                    ]
                )
            else:
                embedding = torch.cat([span_1.tokens[0].get_embedding(), span_2.tokens[0].get_embedding()])

            relation_embeddings.append(embedding)
        if relation_embeddings:
            embedding_tensor = torch.stack(relation_embeddings)
        else:
            embedding_tensor = torch.zeros(0, self.relation_representation_length)

        return (embedding_tensor,)

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:

            eval_line = f"\n{datapoint.to_original_text()}\n"

            for relation in datapoint.get_relations(gold_label_type):
                symbol = (
                    "✓" if relation.get_label(gold_label_type).value == relation.get_label("predicted").value else "❌"
                )
                eval_line += (
                    f' - "{relation.text}"\t{relation.get_label(gold_label_type).value}'
                    f' --> {relation.get_label("predicted").value} ({symbol})\n'
                )

            lines.append(eval_line)
        return lines

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_type": self.entity_label_type,
            "weight_dict": self.weight_dict,
            "pooling_operation": self.pooling_operation,
            "entity_pair_filters": self.entity_pair_filters,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            entity_label_type=state.get("entity_label_type"),
            loss_weights=state.get("weight_dict"),
            pooling_operation=state.get("pooling_operation"),
            entity_pair_filters=state.get("entity_pair_filters"),
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["relations"] = "/".join([hu_path, "relations", "relations-v11.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name


def create_position_string(head: Span, tail: Span) -> str:
    return f"{head.unlabeled_identifier} -> {tail.unlabeled_identifier}"
