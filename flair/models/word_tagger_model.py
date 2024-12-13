import logging
from pathlib import Path
from typing import Any, Union

import torch
from deprecated.sphinx import deprecated

import flair.nn
from flair.data import Dictionary, Sentence, Span, Token
from flair.embeddings import TokenEmbeddings

log = logging.getLogger("flair")


class TokenClassifier(flair.nn.DefaultClassifier[Sentence, Token]):
    """This is a simple class of models that tags individual words in text."""

    def __init__(
        self,
        embeddings: TokenEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        span_encoding: str = "BIOES",
        **classifierargs,
    ) -> None:
        """Initializes a TokenClassifier.

        Args:
            embeddings: word embeddings used in tagger
            label_dictionary: dictionary of labels or BIO/BIOES tags you want to predict
            label_type: string identifier for tag type
            span_encoding: the format to encode spans as tags, either "BIO" or "BIOES"
            **classifierargs: The arguments propagated to :meth:`flair.nn.DefaultClassifier.__init__`
        """
        # if the classifier predicts BIO/BIOES span labels, the internal label dictionary must be computed
        if label_dictionary.span_labels:
            internal_label_dictionary = self._create_internal_label_dictionary(label_dictionary, span_encoding)
        else:
            internal_label_dictionary = label_dictionary

        super().__init__(
            embeddings=embeddings,
            label_dictionary=internal_label_dictionary,
            final_embedding_size=embeddings.embedding_length,
            **classifierargs,
        )

        # fields in case this is a span-prediction problem
        self.span_prediction_problem = self._determine_if_span_prediction_problem(internal_label_dictionary)
        self.span_encoding = span_encoding

        # the label type
        self._label_type: str = label_type

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @staticmethod
    def _create_internal_label_dictionary(label_dictionary, span_encoding):
        internal_label_dictionary = Dictionary(add_unk=False)
        for label in label_dictionary.get_items():
            if label == "<unk>":
                continue
            internal_label_dictionary.add_item("O")
            if span_encoding == "BIOES":
                internal_label_dictionary.add_item("S-" + label)
                internal_label_dictionary.add_item("B-" + label)
                internal_label_dictionary.add_item("E-" + label)
                internal_label_dictionary.add_item("I-" + label)
            if span_encoding == "BIO":
                internal_label_dictionary.add_item("B-" + label)
                internal_label_dictionary.add_item("I-" + label)

        return internal_label_dictionary

    def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
        return any(item.startswith(("B-", "S-", "I-")) for item in dictionary.get_items())

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            **kwargs,
        )

    def _get_embedding_for_data_point(self, prediction_data_point: Token) -> torch.Tensor:
        names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(names)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> list[Token]:
        # special handling during training if this is a span prediction problem
        if self.training and self.span_prediction_problem:
            for token in sentence.tokens:
                token.set_label(self.label_type, "O")
                for span in sentence.get_spans(self.label_type):
                    span_label = span.get_label(self.label_type).value
                    if len(span) == 1:
                        if self.span_encoding == "BIOES":
                            span.tokens[0].set_label(self.label_type, "S-" + span_label)
                        elif self.span_encoding == "BIO":
                            span.tokens[0].set_label(self.label_type, "B-" + span_label)
                    else:
                        for token in span.tokens:
                            token.set_label(self.label_type, "I-" + span_label)
                        span.tokens[0].set_label(self.label_type, "B-" + span_label)
                        if self.span_encoding == "BIOES":
                            span.tokens[-1].set_label(self.label_type, "E-" + span_label)

        return sentence.tokens

    def _post_process_batch_after_prediction(self, batch, label_name):
        if self.span_prediction_problem:
            for sentence in batch:
                # internal variables
                previous_tag = "O-"
                current_span: list[Token] = []

                for token in sentence:
                    bioes_tag = token.get_label(label_name).value

                    # non-set tags are OUT tags
                    if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
                        bioes_tag = "O-"

                    # anything that is not OUT is IN
                    in_span = bioes_tag != "O-"

                    # does this prediction start a new span?
                    starts_new_span = False

                    if bioes_tag[:2] in {"B-", "S-"} or (
                        in_span
                        and previous_tag[2:] != bioes_tag[2:]
                        and (bioes_tag[:2] == "I-" or previous_tag[2:] == "S-")
                    ):
                        # B- and S- always start new spans
                        # if the predicted class changes, I- starts a new span
                        # if the predicted class changes and S- was previous tag, start a new span
                        starts_new_span = True

                    # if an existing span is ended (either by reaching O or starting a new span)
                    if (starts_new_span or not in_span) and len(current_span) > 0:
                        sentence[current_span[0].idx - 1 : current_span[-1].idx].set_label(label_name, previous_tag[2:])
                        # reset for-loop variables for new span
                        current_span = []

                    if in_span:
                        current_span.append(token)

                    # remember previous tag
                    previous_tag = bioes_tag

                    token.remove_labels(label_name)
                    token.remove_labels(self.label_type)

                # if there is a span at end of sentence, add it
                if len(current_span) > 0:
                    sentence[current_span[0].idx - 1 : current_span[-1].idx].set_label(label_name, previous_tag[2:])

    @property
    def label_type(self):
        return self._label_type

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        if self.span_prediction_problem:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                # set predicted token-level
                for predicted_label in datapoint.get_labels("predicted"):
                    predicted_span: Span = predicted_label.data_point
                    prefix = "B-"
                    for token in predicted_span:
                        token.set_label("predicted_bio", prefix + predicted_label.value)
                        prefix = "I-"

                # now print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label('gold_bio').value} "
                        f"{token.get_label('predicted_bio').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")

        else:
            for datapoint in batch:
                # print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label(gold_label_type).value} "
                        f"{token.get_label('predicted').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")
        return lines

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "TokenClassifier":
        from typing import cast

        return cast("TokenClassifier", super().load(model_path=model_path))


@deprecated(reason="The WordTagger was renamed to :class:`flair.models.TokenClassifier`.", version="0.12.2")
class WordTagger(TokenClassifier):
    pass
