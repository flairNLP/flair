import logging
from pathlib import Path
from typing import List

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence
from flair.embeddings import Embeddings
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class TextClassifier(flair.nn.DefaultClassifier[Sentence, Sentence]):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text
    representation, and puts the text representation in the end into a linear
    layer to get the actual class label. The model can handle single and multi
    class data sets.
    """

    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(TextClassifier, self).__init__(
            **classifierargs,
            final_embedding_size=document_embeddings.embedding_length,
        )

        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        self._label_type = label_type

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _embed_prediction_data_point(self, prediction_data_point: Sentence) -> torch.Tensor:
        embedding_names = self.document_embeddings.get_names()
        return prediction_data_point.get_embedding(embedding_names)

    def _get_prediction_data_points(self, sentences: List[Sentence]) -> List[Sentence]:
        return sentences

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "weight_dict": self.weight_dict,
        }
        return model_state

    @property
    def _inner_embeddings(self) -> Embeddings[Sentence]:
        return self.document_embeddings

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        return super()._init_model_with_state_dict(
            state,
            document_embeddings=state.get("document_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            multi_label=state.get("multi_label"),
            multi_label_threshold=state.get("multi_label_threshold", 0.5),
            loss_weights=state.get("weight_dict"),
            **kwargs,
        )

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["de-offensive-language"] = "/".join(
            [hu_path, "de-offensive-language", "germ-eval-2018-task-1-v0.8.pt"]
        )

        # English sentiment models
        model_map["sentiment"] = "/".join(
            [
                hu_path,
                "sentiment-curated-distilbert",
                "sentiment-en-mix-distillbert_4.pt",
            ]
        )
        model_map["en-sentiment"] = "/".join(
            [
                hu_path,
                "sentiment-curated-distilbert",
                "sentiment-en-mix-distillbert_4.pt",
            ]
        )
        model_map["sentiment-fast"] = "/".join(
            [hu_path, "sentiment-curated-fasttext-rnn", "sentiment-en-mix-ft-rnn_v8.pt"]
        )

        # Communicative Functions Model
        model_map["communicative-functions"] = "/".join([hu_path, "comfunc", "communicative-functions.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    @property
    def label_type(self):
        return self._label_type
