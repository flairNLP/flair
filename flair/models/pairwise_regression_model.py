from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.embeddings
import flair.nn
from flair.data import Corpus, Dictionary, Sentence, TextPair, _iter_dataset
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.nn.model import ReduceTransformerVocabMixin
from flair.training_utils import EmbeddingStorageMode, MetricRegression, Result, store_embeddings


class TextPairRegressor(flair.nn.Model[TextPair], ReduceTransformerVocabMixin):
    """Text Pair Regression Model for tasks such as Semantic Textual Similarity Benchmark.

    The model takes document embeddings and puts resulting text representation(s) into a linear layer to get the
    score. We provide two ways to embed the DataPairs: Either by embedding both DataPoints
    and concatenating the resulting vectors ("embed_separately=True") or by concatenating the DataPoints and embedding
    the resulting vector ("embed_separately=False").
    """

    def __init__(
        self,
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        embed_separately: bool = False,
        dropout: float = 0.0,
        locked_dropout: float = 0.0,
        word_dropout: float = 0.0,
        decoder: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialize the Text Pair Regression Model.

        Args:
            label_type: name of the label
            embed_separately: if True, the sentence embeddings will be concatenated,
              if False both sentences will be combined and newly embedded.
            dropout: dropout
            locked_dropout: locked_dropout
            word_dropout:  word_dropout
            decoder: if provided, a that specific layer will be used as decoder,
              otherwise a linear layer with random parameters will be created.
            embeddings: embeddings used to embed each data point
        """
        super().__init__()

        self.embeddings: flair.embeddings.DocumentEmbeddings = embeddings
        self.label_name = label_type
        self.embed_separately = embed_separately

        if not self.embed_separately:
            # set separator to concatenate two sentences
            self.sep = " "
            if isinstance(
                self.embeddings,
                flair.embeddings.document.TransformerDocumentEmbeddings,
            ):
                if self.embeddings.tokenizer.sep_token:
                    self.sep = " " + str(self.embeddings.tokenizer.sep_token) + " "
                else:
                    self.sep = " [SEP] "

        self.decoder: torch.nn.Module
        if decoder is None:
            self.decoder = nn.Linear(
                2 * embeddings.embedding_length if embed_separately else embeddings.embedding_length, 1
            )
            nn.init.xavier_uniform_(self.decoder.weight)

        else:
            self.decoder = decoder

        # init dropouts
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
        self.word_dropout = flair.nn.WordDropout(word_dropout)

        self.loss_function = nn.MSELoss(reduction="sum")

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self.label_name

    def get_used_tokens(
        self, corpus: Corpus, context_length: int = 0, respect_document_boundaries: bool = True
    ) -> Iterable[list[str]]:
        for sentence_pair in _iter_dataset(corpus.get_all_sentences()):
            yield [t.text for t in sentence_pair.first]
            yield [t.text for t in sentence_pair.first.left_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.first.right_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.second]
            yield [t.text for t in sentence_pair.second.left_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.second.right_context(context_length, respect_document_boundaries)]

    def forward_loss(self, pairs: list[TextPair]) -> tuple[torch.Tensor, int]:
        loss, num = self._forward_loss_and_scores(pairs=pairs, return_num=True, return_scores=False)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(num, int)

        return loss, num

    def _forward_loss_and_scores(self, pairs: list[TextPair], return_num=True, return_scores=True) -> tuple:
        # make a forward pass to produce embedded data points and labels
        pairs = [pair for pair in pairs if self._filter_data_point(pair)]

        if len(pairs) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        target_tensor = self._prepare_target_tensor(pairs)

        if target_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(pairs)

        # decode
        scores = self.decoder(data_point_tensor)[:, 0]

        # calculate the loss
        loss, num = self._calculate_loss(scores, target_tensor)

        return_value: tuple[Any, ...] = (loss,)

        if return_num:
            return_value += (num,)

        if return_scores:
            return_value += (scores,)

        return return_value

    def _calculate_loss(self, scores: torch.Tensor, target_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.loss_function(scores, target_tensor), target_tensor.size(0)

    def _prepare_target_tensor(self, pairs: list[TextPair]):
        target_values = [
            torch.tensor([float(label.value) for label in pair.get_labels(self.label_name)], dtype=torch.float)
            for pair in pairs
        ]

        return torch.cat(target_values, 0).to(flair.device)

    def _filter_data_point(self, pair: TextPair) -> bool:
        return len(pair) > 0

    def _encode_data_points(self, data_points: list[TextPair]) -> torch.Tensor:
        # get a tensor of data points
        data_point_tensor = torch.stack([self._get_embedding_for_data_point(data_point) for data_point in data_points])

        # do dropout
        data_point_tensor = data_point_tensor.unsqueeze(1)
        data_point_tensor = self.dropout(data_point_tensor)
        data_point_tensor = self.locked_dropout(data_point_tensor)
        data_point_tensor = self.word_dropout(data_point_tensor)
        data_point_tensor = data_point_tensor.squeeze(1)

        return data_point_tensor

    def _get_embedding_for_data_point(self, prediction_data_point: TextPair) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        if self.embed_separately:
            self.embeddings.embed([prediction_data_point.first, prediction_data_point.second])
            return torch.cat(
                [
                    prediction_data_point.first.get_embedding(embedding_names),
                    prediction_data_point.second.get_embedding(embedding_names),
                ],
                0,
            )
        else:
            # If the concatenated version of the text pair does not exist yet, create it
            if prediction_data_point.concatenated_data is None:
                concatenated_sentence = Sentence(
                    prediction_data_point.first.to_tokenized_string()
                    + self.sep
                    + prediction_data_point.second.to_tokenized_string(),
                    use_tokenizer=False,
                )
                prediction_data_point.concatenated_data = concatenated_sentence
            self.embeddings.embed(prediction_data_point.concatenated_data)
            return prediction_data_point.concatenated_data.get_embedding(embedding_names)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_type": self.label_type,
            "embed_separately": self.embed_separately,
            "dropout": self.dropout.p,
            "word_dropout": self.word_dropout.dropout_rate,
            "locked_dropout": self.locked_dropout.dropout_rate,
            "decoder": self.decoder,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state: dict[str, Any], **kwargs):
        """Initializes a TextPairRegressor model from a state dictionary (exported by _get_state_dict).

        Requires keys 'state_dict', 'document_embeddings', and 'label_type' in the state dictionary.
        """
        if "document_embeddings" in state:
            state["embeddings"] = state.pop("document_embeddings")  # need to rename this parameter
        # add Model arguments
        for arg in [
            "embeddings",
            "label_type",
            "embed_separately",
            "dropout",
            "word_dropout",
            "locked_dropout",
            "decoder",
        ]:
            if arg not in kwargs and arg in state:
                kwargs[arg] = state[arg]

        return super()._init_model_with_state_dict(state, **kwargs)

    def predict(
        self,
        pairs: Union[TextPair, list[TextPair]],
        mini_batch_size: int = 32,
        verbose: bool = False,
        label_name: Optional[str] = None,
        embedding_storage_mode="none",
    ) -> list[TextPair]:
        if label_name is None:
            label_name = self.label_name if self.label_name is not None else "label"

        with torch.no_grad():
            if isinstance(pairs, list):
                if len(pairs) == 0:
                    return []
            else:
                pairs = [pairs]

            filtered_pairs = [pair for pair in pairs if self._filter_data_point(pair)]

            if len(filtered_pairs) == 0:
                return pairs

            reordered_pairs = sorted(filtered_pairs, key=lambda pair: len(pair.first) + len(pair.second), reverse=True)

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(reordered_pairs),
                batch_size=mini_batch_size,
            )
            # progress bar for verbosity
            if verbose:
                progress_bar = tqdm(dataloader)
                progress_bar.set_description("Batch inference")
                dataloader = progress_bar

            for batch in dataloader:
                # stop if all sentences are empty
                if not batch:
                    continue

                data_point_tensor = self._encode_data_points(batch)
                scores = self.decoder(data_point_tensor)

                for sentence, score in zip(batch, scores.tolist()):
                    sentence.set_label(label_name, value=str(score[0]))

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            return pairs

    def evaluate(
        self,
        data_points: Union[list[TextPair], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path, None] = None,
        embedding_storage_mode: EmbeddingStorageMode = "none",
        mini_batch_size: int = 32,
        main_evaluation_metric: tuple[str, str] = ("correlation", "pearson"),
        exclude_labels: Optional[list[str]] = None,
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        exclude_labels = exclude_labels if exclude_labels is not None else []
        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)
        data_loader = DataLoader(data_points, batch_size=mini_batch_size)

        with torch.no_grad():
            eval_loss = torch.zeros(1, device=flair.device)

            metric = MetricRegression("Evaluation")

            if out_path is not None:
                out_file = open(out_path, "w", encoding="utf-8")  # noqa: SIM115

            total_count = 0

            try:
                for batch in data_loader:
                    if isinstance(batch, Sentence):
                        batch = [batch]

                    loss, num, scores = self._forward_loss_and_scores(batch, return_scores=True)

                    true_values = []
                    for sentence in batch:
                        total_count += 1
                        for label in sentence.get_labels(gold_label_type):
                            true_values.append(float(label.value))

                    results = scores.cpu().tolist()

                    eval_loss += loss

                    metric.true.extend(true_values)
                    metric.pred.extend(results)

                    if out_path is not None:
                        for pair, prediction, true_value in zip(batch, results, true_values):
                            eval_line = f"{pair.first.to_original_text()}\t{pair.second.to_original_text()}\t{true_value}\t{prediction}\n"
                            out_file.write(eval_line)

                    store_embeddings(batch, embedding_storage_mode)
            finally:
                if out_path is not None:
                    out_file.close()

            eval_loss /= total_count

            detailed_result = (
                f"AVG: mse: {metric.mean_squared_error():.4f} - "
                f"mae: {metric.mean_absolute_error():.4f} - "
                f"pearson: {metric.pearsonr():.4f} - "
                f"spearman: {metric.spearmanr():.4f}"
            )

            scores = {
                "loss": eval_loss.item(),
                "mse": metric.mean_squared_error(),
                "mae": metric.mean_absolute_error(),
                "pearson": metric.pearsonr(),
                "spearman": metric.spearmanr(),
            }

            if main_evaluation_metric[0] in ("correlation", "other"):
                main_score = scores[main_evaluation_metric[1]]
            else:
                main_score = scores["spearman"]

            return Result(
                main_score=main_score,
                detailed_results=detailed_result,
                scores=scores,
            )
