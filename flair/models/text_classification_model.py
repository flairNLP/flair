import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence
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
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param embeddings: embeddings used to embed each data point
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
            embeddings=embeddings,
            final_embedding_size=embeddings.embedding_length,
        )

        self._label_type = label_type

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _get_embedding_for_data_point(self, prediction_data_point: Sentence) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(embedding_names)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> List[Sentence]:
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
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "TextClassifier":
        from typing import cast

        return cast("TextClassifier", super().load(model_path=model_path))



class TextClassifierLossModifications(TextClassifier):
    def __init__(
        self,
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        loss: str,
        batch_avg: bool,
        **classifierargs,
    ):
        super(TextClassifierLossModifications, self).__init__(
                embeddings=embeddings,
                label_type=label_type,
                **classifierargs,
            )
        self.pe_norm = False # cce * pe
        self.batch_avg = batch_avg 
        self.entropy_loss = False # cce + pe

        if loss == 'pe_norm':
            self.pe_norm = True # cce * pe
        elif loss == 'entropy_loss':
            self.entropy_loss = True

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # get softmax values
        softmax = torch.nn.functional.softmax(scores)
        # calc entropy for each data point
        entropy = -torch.sum(torch.mul(softmax, torch.log(softmax)), dim = -1)
        # calc cross entropy for each data point
        cross_entropy = torch.nn.functional.nll_loss(torch.log(softmax), labels, reduction='none')
        #log.info(self.pe_norm)
        #log.info(self.model_card["training_parameters"]["epoch"])
        if self.entropy_loss:
            loss = cross_entropy + entropy
        elif self.pe_norm and self.model_card["training_parameters"]["epoch"]>2:
            pe_norm = (entropy.clone().detach()) / (1.5)
            pred=torch.argmax(softmax, dim=1)
            incorrect_pred_indicator = (pred != labels)
            selective_factor = torch.sub(torch.zeros_like(pe_norm).fill_(1), (torch.sub(torch.zeros_like(pe_norm).fill_(1),pe_norm)) * incorrect_pred_indicator)
            loss = torch.mul(cross_entropy, selective_factor)
        else:
            loss = cross_entropy
        if self.batch_avg and self.model_card["training_parameters"]["epoch"]>3:
            # set loss weight for each data point
            batch_weights = torch.where((cross_entropy > 2 * torch.mean(cross_entropy)) & (entropy < torch.mean(entropy)), 0, 1)
            # multiply by loss weight and sum
            loss = torch.mul(loss, batch_weights)
        return torch.sum(loss), labels.size(0)