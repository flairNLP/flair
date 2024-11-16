import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Relation, Sentence
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class RelationExtractor(flair.nn.DefaultClassifier[Sentence, Relation]):
    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
        label_type: str,
        entity_label_type: str,
        entity_pair_filters: Optional[list[tuple[str, str]]] = None,
        pooling_operation: str = "first_last",
        train_on_gold_pairs_only: bool = False,
        **classifierargs,
    ) -> None:
        """Initializes a RelationClassifier.

        Args:
            embeddings: embeddings used to embed each data point
            label_type: name of the label
            entity_label_type: name of the labels used to represent entities
            entity_pair_filters: if provided, only classify pairs that apply the filter
            pooling_operation: either "first" or "first_last" how the embeddings of the entities
              should be used to create relation embeddings
            train_on_gold_pairs_only: if True, relations with "O" (no relation) label will be ignored in training.
            **classifierargs: The arguments propagated to :meth:`flair.nn.DefaultClassifier.__init__`
        """
        # pooling operation to get embeddings for entites
        self.pooling_operation = pooling_operation
        relation_representation_length = 2 * embeddings.embedding_length
        if self.pooling_operation == "first_last":
            relation_representation_length *= 2
        super().__init__(
            embeddings=embeddings,
            final_embedding_size=relation_representation_length,
            **classifierargs,
        )

        # set embeddings
        self.embeddings: flair.embeddings.TokenEmbeddings = embeddings

        # set relation and entity label types
        self._label_type = label_type
        self.entity_label_type = entity_label_type
        self.train_on_gold_pairs_only = train_on_gold_pairs_only

        # whether to use gold entity pairs, and whether to filter entity pairs by type
        if entity_pair_filters is not None:
            self.entity_pair_filters: Optional[set[tuple[str, str]]] = set(entity_pair_filters)
        else:
            self.entity_pair_filters = None

        self.to(flair.device)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> list[Relation]:
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

                relation = Relation(span_1, span_2)
                if self.training and self.train_on_gold_pairs_only and relation.get_label(self.label_type).value == "O":
                    continue
                entity_pairs.append(relation)
        return entity_pairs

    def _get_embedding_for_data_point(self, prediction_data_point: Relation) -> torch.Tensor:
        span_1 = prediction_data_point.first
        span_2 = prediction_data_point.second
        embedding_names = self.embeddings.get_names()

        if self.pooling_operation == "first_last":
            return torch.cat(
                [
                    span_1.tokens[0].get_embedding(embedding_names),
                    span_1.tokens[-1].get_embedding(embedding_names),
                    span_2.tokens[0].get_embedding(embedding_names),
                    span_2.tokens[-1].get_embedding(embedding_names),
                ]
            )
        else:
            return torch.cat(
                [span_1.tokens[0].get_embedding(embedding_names), span_2.tokens[0].get_embedding(embedding_names)]
            )

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
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_type": self.entity_label_type,
            "weight_dict": self.weight_dict,
            "pooling_operation": self.pooling_operation,
            "entity_pair_filters": self.entity_pair_filters,
            "train_on_gold_pairs_only": self.train_on_gold_pairs_only,
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
            train_on_gold_pairs_only=state.get("train_on_gold_pairs_only", False),
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

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "RelationExtractor":
        from typing import cast

        return cast("RelationExtractor", super().load(model_path=model_path))
