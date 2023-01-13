import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

import flair.embeddings
import flair.nn
from flair.data import DT, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.file_utils import cached_path
from flair.training_utils import store_embeddings

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
            "document_embeddings": self.embeddings,
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


class TextClassifierProbes(TextClassifier):
    """
    Text Classification Model with Linear Classifier Probes and the ability to output Prediction Depth.
    """

    def __init__(
        self,
        embeddings: flair.embeddings.TransformerDocumentEmbeddings,  # 'layers' param of embeddings class has to be set to 'all'
        label_type: str,
        **classifierargs,
    ):

        super(TextClassifierProbes, self).__init__(
            embeddings=embeddings,
            label_type=label_type,
            **classifierargs,
        )

        self.n_layers = len(
            embeddings.layer_indexes
        )  # the output of the emb layer before the transformer blocks counts as well
        self.final_embedding_size = int(embeddings.embedding_length / self.n_layers)
        self.decoder = flair.nn.LayerwiseDecoder(
            n_layers=self.n_layers, emb_size=self.final_embedding_size, n_labels=len(self.label_dictionary)
        )

        self.to(flair.device)

    def _encode_data_points(self, sentences, data_points):

        # embed sentences
        self.embeddings.embed(sentences)

        # get a tensor of data points
        data_point_tensor = torch.stack([self._get_embedding_for_data_point(data_point) for data_point in data_points])

        # reshape & transpose
        data_point_tensor = torch.reshape(
            data_point_tensor,
            (data_point_tensor.size(0), self.n_layers, int(data_point_tensor.size(1) / self.n_layers)),
        )
        data_point_tensor = torch.transpose(data_point_tensor, 0, 1)

        # do dropout
        for i in range(self.n_layers):

            layer_tensor = data_point_tensor[i]
            layer_tensor = layer_tensor.unsqueeze(1)
            layer_tensor = self.dropout(layer_tensor)
            layer_tensor = self.locked_dropout(layer_tensor)
            layer_tensor = self.word_dropout(layer_tensor)
            layer_tensor = layer_tensor.squeeze(1)
            data_point_tensor[i] = layer_tensor

        return data_point_tensor

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:

        layer_weights = torch.arange(1, self.n_layers+1, device=flair.device)
        layer_weighted_loss = 0

        for i in range(self.n_layers):
            layer_loss = self.loss_function(scores[i], labels)
            layer_weighted_loss += layer_weights[i] * layer_loss

        weighted_average_loss = layer_weighted_loss / sum(layer_weights)

        return weighted_average_loss, labels.size(0)

    def _calculate_pd(self, scores: torch.Tensor, label_threshold=None) -> int:
        """
        Calculates the prediction depth for a given (single) data point.
        :param scores: tensor with softmax or sigmoid scores of all layers
        :param label_threshold: relevant only for multi-label classification
        """
        pd = self.n_layers - 1

        if self.multi_label:
            for i in range(self.n_layers - 2, -1, -1):
                if scores[i] > label_threshold:
                    pd -= 1
                else:
                    break

        else:
            pred_labels = torch.argmax(scores, dim=-1)
            for i in range(self.n_layers - 2, -1, -1):  # iterate over the layers starting from the penultimate one
                if pred_labels[i] == pred_labels[-1]:
                    pd -= 1
                else:
                    break

        return pd

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Copy of the DefaultClassifier's predict method with adjustments necessary to facilitate PD calculation.
        The adjusted lines are commented with 'PD' keyword for visibitility.
        The calculated PD is added to a data point directly as a label.
        """
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            reordered_sentences = self._sort_data(sentences)

            if len(reordered_sentences) == 0:
                return sentences

            if len(reordered_sentences) > mini_batch_size:
                batches: Union[DataLoader, List[List[DT]]] = DataLoader(
                    dataset=FlairDatapointDataset(reordered_sentences),
                    batch_size=mini_batch_size,
                )
                # progress bar for verbosity
                if verbose:
                    progress_bar = tqdm(batches)
                    progress_bar.set_description("Batch inference")
                    batches = progress_bar
            else:
                batches = [reordered_sentences]

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0
            for batch in batches:

                # filter data points in batch
                batch = [dp for dp in batch if self._filter_data_point(dp)]

                # stop if all sentences are empty
                if not batch:
                    continue

                data_points = self._get_data_points_for_batch(batch)

                if not data_points:
                    continue

                # pass data points through network and decode
                data_point_tensor = self._encode_data_points(batch, data_points)
                scores = self.decoder(data_point_tensor)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    for sentence in data_points:
                        sentence.remove_labels(label_name)

                    if return_loss:
                        gold_labels = self._prepare_label_tensor(data_points)
                        overall_loss += self._calculate_loss(scores, gold_labels)[0]
                        label_count += len(data_points)

                    if self.multi_label:
                        sigmoided = torch.sigmoid(scores)  # size: (n_sentences, n_classes)
                        n_labels = sigmoided.size(-1)  # adjust indexing for PD
                        for s_idx, data_point in enumerate(data_points):
                            for l_idx in range(n_labels):
                                label_value = self.label_dictionary.get_item_for_index(l_idx)
                                if label_value == "O":
                                    continue
                                label_threshold = self._get_label_threshold(label_value)
                                label_score = sigmoided[-1, s_idx, l_idx].item()  # adjust indexing for PD
                                # add PD label
                                if label_score > label_threshold:
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                                    pd = self._calculate_pd(sigmoided[:, s_idx, l_idx], label_threshold)
                                    data_point.add_label(typename="PD", value="PD_" + label_value, score=pd)
                                elif return_probabilities_for_all_classes:
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)

                    else:
                        softmax = torch.nn.functional.softmax(scores, dim=-1)

                        if return_probabilities_for_all_classes:
                            n_labels = softmax.size(-1)  # adjust indexing for PD
                            for s_idx, data_point in enumerate(data_points):
                                # add PD label
                                pd = self._calculate_pd(softmax[:, s_idx, :])
                                data_point.add_label(typename="PD", value="PD", score=pd)
                                for l_idx in range(n_labels):
                                    label_value = self.label_dictionary.get_item_for_index(l_idx)
                                    if label_value == "O":
                                        continue
                                    label_score = softmax[-1, s_idx, l_idx].item()  # adjust indexing for PD
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                        else:
                            conf, idx = torch.max(softmax, dim=-1)
                            conf, idx = conf[-1], idx[-1]  # adjust indexing for PD
                            for j, (data_point, c, i) in enumerate(
                                zip(data_points, conf, idx)
                            ):  # enumerate added for PD
                                label_value = self.label_dictionary.get_item_for_index(i.item())
                                if label_value == "O":
                                    continue
                                data_point.add_label(typename=label_name, value=label_value, score=c.item())
                                # add PD label
                                pd = self._calculate_pd(softmax[:, j, :])
                                data_point.add_label(typename="PD", value="PD", score=pd)

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count
