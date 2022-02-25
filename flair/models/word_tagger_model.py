import logging
from typing import List, Union

import torch
import torch.nn

import flair.nn
from flair.data import Dictionary, Sentence
from flair.embeddings import TokenEmbeddings

log = logging.getLogger("flair")


class WordTagger(flair.nn.DefaultClassifier[Sentence]):
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
            label_dictionary=tag_dictionary, final_embedding_size=embeddings.embedding_length, **classifierargs
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
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            **kwargs,
        )

    def forward_pass(
        self,
        sentences: Union[List[Sentence], Sentence],
        for_prediction: bool = False,
    ):
        if not isinstance(sentences, list):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        # get all tokens in this mini-batch
        all_tokens = [token for sentence in sentences for token in sentence]

        all_embeddings = [token.get_embedding(names) for token in all_tokens]

        embedded_tokens = torch.stack(all_embeddings)

        labels = [[token.get_label(self.label_type).value] for token in all_tokens]

        if for_prediction:
            return embedded_tokens, labels, all_tokens

        return embedded_tokens, labels

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
