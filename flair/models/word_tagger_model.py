import logging
from typing import List, Union

import torch
import torch.nn

import flair.nn
from flair.data import Dictionary, Label, Sentence
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
        super().__init__(label_dictionary=tag_dictionary, **classifierargs)

        # embeddings
        self.embeddings = embeddings

        # dictionaries
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        # linear layer
        self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "tag_dictionary": self.label_dictionary,
            "tag_type": self.tag_type,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = WordTagger(
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def forward_pass(
        self,
        sentences: Union[List[Sentence], Sentence],
        return_label_candidates: bool = False,
    ):
        if not isinstance(sentences, list):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        # get all tokens in this mini-batch
        all_tokens = [token for sentence in sentences for token in sentence]

        all_embeddings = [token.get_embedding(names) for token in all_tokens]

        embedding_tensor = torch.stack(all_embeddings)

        scores = self.linear(embedding_tensor)

        labels = [[token.get_tag(self.label_type).value] for token in all_tokens]

        if return_label_candidates:
            empty_label_candidates = [Label(value=None, score=0.0) for token in all_tokens]
            return scores, labels, all_tokens, empty_label_candidates

        return scores, labels

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
                    f"{token.get_tag(gold_label_type, 'O').value} "
                    f"{token.get_tag('predicted', 'O').value}\n"
                )
                lines.append(eval_line)
            lines.append("\n")
        return lines
