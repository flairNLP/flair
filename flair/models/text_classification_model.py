import logging
from pathlib import Path
from typing import Any, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class TextClassifier(flair.nn.DefaultClassifier[Sentence, Sentence]):
    """Text Classification Model.

    The model takes word embeddings, puts them into an RNN to obtain a text
    representation, and puts the text representation in the end into a linear
    layer to get the actual class label. The model can handle single and multi
    class data sets.
    """

    def __init__(
        self,
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        **classifierargs,
    ) -> None:
        """Initializes a TextClassifier.

        Args:
            embeddings: embeddings used to embed each data point
            label_dictionary: dictionary of labels you want to predict
            label_type: string identifier for tag type
            multi_label: auto-detected by default, but you can set this to True to force multi-label predictions
                or False to force single-label predictions.
            multi_label_threshold: If multi-label you can set the threshold to make predictions
            beta: Parameter for F-beta score for evaluation and training annealing
            loss_weights: Dictionary of weights for labels for the loss function. If any label's weight is
                unspecified it will default to 1.0
            **classifierargs: The arguments propagated to :meth:`flair.nn.DefaultClassifier.__init__`
        """
        super().__init__(
            **classifierargs,
            embeddings=embeddings,
            final_embedding_size=embeddings.embedding_length,
        )

        self._label_type = label_type

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _get_embedding_for_data_point(self, prediction_data_point: Sentence) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(embedding_names)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> list[Sentence]:
        return [sentence]

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "weight_dict": self.weight_dict,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        import re

        # remap state dict for models serialized with Flair <= 0.11.3
        state_dict = state["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[re.sub("^document_embeddings\\.", "embeddings.", key)] = state_dict.pop(key)

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("document_embeddings"),
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

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "TextClassifier":
        from typing import cast

        return cast("TextClassifier", super().load(model_path=model_path))
