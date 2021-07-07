import warnings
from collections import Counter
from pathlib import Path

import torch.nn

from abc import abstractmethod

from typing import Union, List, Tuple, Optional

from torch import Tensor
from torch.utils.data.dataset import Dataset

import flair
from flair import file_utils
from flair.data import DataPoint, Sentence, Dictionary
from flair.datasets import DataLoader, SentenceDataset
from flair.training_utils import Result, store_embeddings


class Model(torch.nn.Module):
    """Abstract base class for all downstream task models in Flair, such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    @abstractmethod
    def forward_loss(
            self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""
        pass

    @abstractmethod
    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            mini_batch_size: int,
            num_workers: int,
            out_path: Path = None,
            embedding_storage_mode: str = "none",
            main_evaluation_metric: Tuple[str, str] = ("micro avg", 'f1-score'),
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        pass

    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""
        pass

    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        pass

    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load(cls, model: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location='cpu')

        model = cls._init_model_with_state_dict(state)

        model.eval()
        model.to(flair.device)

        return model


class Classifier(Model):

    def evaluate_classification(
            self,
            sentences: Union[List[Sentence], Dataset],
            gold_label_type: str,
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
            exclude_labels: List[str] = [],
    ) -> Result:
        import numpy as np
        import sklearn

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        with torch.no_grad():

            eval_loss = 0
            average_over = 0

            lines: List[str] = []

            all_spans: List[str] = []
            true_values = {}
            predictions = {}

            sentence_id = 0
            for batch in data_loader:

                # remove any previously predicted labels
                for sentence in batch:
                    sentence.remove_labels('predicted')

                # predict for batch
                loss_and_count = self.predict(batch,
                                              embedding_storage_mode=embedding_storage_mode,
                                              mini_batch_size=mini_batch_size,
                                              label_name='predicted',
                                              return_loss=True)

                if isinstance(loss_and_count, Tuple):
                    average_over += loss_and_count[1]
                    eval_loss += loss_and_count[0]
                else:
                    eval_loss += loss_and_count

                # get the gold labels
                for sentence in batch:
                    for gold_label in sentence.get_labels(gold_label_type):
                        representation = str(sentence_id) + ': ' + gold_label.identifier
                        true_values[representation] = gold_label.value
                        if representation not in all_spans:
                            all_spans.append(representation)

                    for predicted_span in sentence.get_labels("predicted"):
                        representation = str(sentence_id) + ': ' + predicted_span.identifier
                        predictions[representation] = predicted_span.value
                        if representation not in all_spans:
                            all_spans.append(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

            #     for sentence in batch:
            #         for token in sentence:
            #             eval_line = f"{token.text} {token.get_tag(label_type).value} {token.get_tag('predicted').value}\n"
            #             lines.append(eval_line)
            #         lines.append("\n")
            #
            # # write predictions to out_file if set
            # if out_path:
            #     with open(Path(out_path), "w", encoding="utf-8") as outfile:
            #         outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for label in true_values.values():
                evaluation_label_dictionary.add_item(label)
            for label in predictions.values():
                evaluation_label_dictionary.add_item(label)

            # finally, compute numbers
            y_true = []
            y_pred = []

            for span in all_spans:

                true_value = true_values[span] if span in true_values else 'O'
                prediction = predictions[span] if span in predictions else 'O'

                true_idx = evaluation_label_dictionary.get_idx_for_item(true_value)
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for i in range(len(evaluation_label_dictionary)):
                    y_true_instance[true_idx] = 1
                y_true.append(y_true_instance.tolist())

                pred_idx = evaluation_label_dictionary.get_idx_for_item(prediction)
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for i in range(len(evaluation_label_dictionary)):
                    y_pred_instance[pred_idx] = 1
                y_pred.append(y_pred_instance.tolist())

        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter()
        counter.update(true_values.values())
        counter.update(predictions.values())

        for label_name, count in counter.most_common():
            if label_name == 'O': continue
            if label_name in exclude_labels: continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        classification_report = sklearn.metrics.classification_report(
            y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=labels,
        )

        classification_report_dict = sklearn.metrics.classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True, labels=labels,
        )

        accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)

        precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
        recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
        micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
        macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {micro_f_score}"
                f"\n- F-score (macro) {macro_f_score}"
                f"\n- Accuracy {accuracy_score}"
                "\n\nBy class:\n" + classification_report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        result = Result(
            main_score=classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]],
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss
        )

        return result


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)
