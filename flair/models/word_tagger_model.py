import logging
from typing import List

import torch

import flair.nn
from flair.data import Dictionary, Sentence, Token
from flair.embeddings import Embeddings, TokenEmbeddings

log = logging.getLogger("flair")


class WordTagger(flair.nn.DefaultClassifier[Sentence, Token]):
    """
    This is a simple class of models that tags individual words in text.
    """

    def __init__(
        self,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        **classifierargs,
    ):
        """
        Initializes a WordTagger
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """
        super().__init__(
            label_dictionary=tag_dictionary,
            final_embedding_size=embeddings.embedding_length,
            **classifierargs,
        )

        # embeddings
        self.embeddings = embeddings

        # dictionaries
        self.tag_type: str = tag_type

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings,
            "tag_dictionary": self.label_dictionary,
            "tag_type": self.tag_type,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            tag_dictionary=state.get("tag_dictionary"),
            tag_type=state.get("tag_type"),
            **kwargs,
        )

    @property
    def _inner_embeddings(self) -> Embeddings[Sentence]:
        return self.embeddings

    def _embed_prediction_data_point(self, prediction_data_point: Token) -> torch.Tensor:
        names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(names)

    def _get_prediction_data_points(self, sentences: List[Sentence]) -> List[Token]:
        tokens: List[Token] = []

        for sentence in sentences:
            tokens.extend(sentence)
        return tokens

    @property
    def label_type(self):
        return self.tag_type

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # now print labels in CoNLL format
            for token in datapoint:
                eval_line = (
                    f"{token.text} "
                    f"{token.get_label(gold_label_type, 'O').value} "
                    f"{token.get_label('predicted', 'O').value}\n"
                )
                lines.append(eval_line)
            lines.append("\n")
        return lines
