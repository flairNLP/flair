import logging
from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import flair
import flair.embeddings
from flair.data import Sentence, Label, DataPoint
from flair.datasets import DataLoader, SentenceDataset
from flair.training_utils import MetricRegression, Result, store_embeddings

log = logging.getLogger("flair")


class TextRegressor(flair.nn.Model):

    def __init__(self, document_embeddings: flair.embeddings.DocumentEmbeddings, label_name: str = 'label'):

        super().__init__()
        log.info("Using REGRESSION - experimental")

        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings
        self.label_name = label_name

        self.decoder = nn.Linear(self.document_embeddings.embedding_length, 1)

        nn.init.xavier_uniform_(self.decoder.weight)

        self.loss_function = nn.MSELoss()

        # auto-spawn on GPU if available
        self.to(flair.device)

    def label_type(self):
        return self.label_name

    def forward(self, sentences):

        self.document_embeddings.embed(sentences)

        embedding_names = self.document_embeddings.get_names()

        text_embedding_list = [sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        scores = self.forward(data_points)

        return self._calculate_loss(scores, data_points)

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.tensor(
                [float(label.value) for label in sentence.labels], dtype=torch.float
            )
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(flair.device)

        return vec

    def predict(
            self,
            sentences: Union[Sentence, List[Sentence]],
            label_name: Optional[str] = None,
            mini_batch_size: int = 32,
            embedding_storage_mode="none",
    ) -> List[Sentence]:

        if label_name == None:
            label_name = self.label_type if self.label_type is not None else 'label'

        with torch.no_grad():
            if type(sentences) is Sentence:
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            # remove previous embeddings
            store_embeddings(filtered_sentences, "none")

            batches = [
                filtered_sentences[x: x + mini_batch_size]
                for x in range(0, len(filtered_sentences), mini_batch_size)
            ]

            for batch in batches:
                scores = self.forward(batch)

                for (sentence, score) in zip(batch, scores.tolist()):
                    sentence.set_label(label_name, value=str(score[0]))

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            return sentences

    def _calculate_loss(
            self, scores: torch.tensor, sentences: List[Sentence]
    ) -> torch.tensor:
        """
        Calculates the loss.
        :param scores: the prediction scores from the model
        :param sentences: list of sentences
        :return: loss value
        """
        return self.loss_function(scores.squeeze(1), self._labels_to_indices(sentences))

    def forward_labels_and_loss(
            self, sentences: Union[Sentence, List[Sentence]]
    ) -> (List[List[float]], torch.tensor):

        scores = self.forward(sentences)
        loss = self._calculate_loss(scores, sentences)
        return scores, loss

    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            **kwargs
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        with torch.no_grad():
            eval_loss = 0

            metric = MetricRegression("Evaluation")

            lines: List[str] = []
            total_count = 0
            for batch_nr, batch in enumerate(data_loader):

                if isinstance(batch, Sentence):
                    batch = [batch]

                scores, loss = self.forward_labels_and_loss(batch)

                true_values = []
                for sentence in batch:
                    total_count += 1
                    for label in sentence.labels:
                        true_values.append(float(label.value))

                results = []
                for score in scores:
                    if type(score[0]) is Label:
                        results.append(float(score[0].score))
                    else:
                        results.append(float(score[0]))

                eval_loss += loss

                metric.true.extend(true_values)
                metric.pred.extend(results)

                for sentence, prediction, true_value in zip(
                        batch, results, true_values
                ):
                    eval_line = "{}\t{}\t{}\n".format(
                        sentence.to_original_text(), true_value, prediction
                    )
                    lines.append(eval_line)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= total_count

            ##TODO: not saving lines yet
            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            log_line = f"{metric.mean_squared_error()}\t{metric.spearmanr()}\t{metric.pearsonr()}"
            log_header = "MSE\tSPEARMAN\tPEARSON"

            detailed_result = (
                f"AVG: mse: {metric.mean_squared_error():.4f} - "
                f"mae: {metric.mean_absolute_error():.4f} - "
                f"pearson: {metric.pearsonr():.4f} - "
                f"spearman: {metric.spearmanr():.4f}"
            )

            result: Result = Result(main_score=metric.pearsonr(),
                                    loss=eval_loss,
                                    log_header=log_header,
                                    log_line=log_line,
                                    detailed_results=detailed_result,
                                    )

            return result

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_name": self.label_type,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        label_name = state["label_name"] if "label_name" in state.keys() else None

        model = TextRegressor(document_embeddings=state["document_embeddings"], label_name=label_name)

        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences
