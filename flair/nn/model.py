import itertools
import logging
import typing
import warnings
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch.nn
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair
from flair import file_utils
from flair.data import DT, DT2, Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import Embeddings
from flair.file_utils import Tqdm
from flair.training_utils import Result, store_embeddings

log = logging.getLogger("flair")


class Model(torch.nn.Module, typing.Generic[DT]):
    """Abstract base class for all downstream task models in Flair,
    such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    model_card: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def label_type(self):
        """Each model predicts labels of a certain type.
        TODO: can we find a better name for this?"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on input tensors for backpropagation or prediction."""
        raise NotImplementedError

    @abstractmethod
    def _prepare_tensors(self, data_points: List[DT]) -> Tuple[torch.Tensor, ...]:
        """Creates the input tensors that will be passed trough the forward call."""
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Performs a forward pass and returns a loss tensor for backpropagation.
        Implement this to enable training."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU  # noqa: E501
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        raise NotImplementedError

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        state_dict = {"state_dict": self.state_dict()}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """Initialize the model from a state dictionary."""
        model = cls(**kwargs)

        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path], checkpoint: bool = False):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        # in Flair <0.9.1, optimizer and scheduler used to train model are not saved
        optimizer = scheduler = None

        # write out a "model card" if one is set
        if self.model_card is not None:

            # special handling for optimizer:
            # remember optimizer class and state dictionary
            if "training_parameters" in self.model_card:
                training_parameters = self.model_card["training_parameters"]

                if "optimizer" in training_parameters:
                    optimizer = training_parameters["optimizer"]
                    if checkpoint:
                        training_parameters["optimizer_state_dict"] = optimizer.state_dict()
                    training_parameters["optimizer"] = optimizer.__class__

                if "scheduler" in training_parameters:
                    scheduler = training_parameters["scheduler"]
                    if checkpoint:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            training_parameters["scheduler_state_dict"] = scheduler.state_dict()
                    training_parameters["scheduler"] = scheduler.__class__

            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

        # restore optimizer and scheduler to model card if set
        if self.model_card is not None:
            if optimizer:
                self.model_card["training_parameters"]["optimizer"] = optimizer
            if scheduler:
                self.model_card["training_parameters"]["scheduler"] = scheduler

    @classmethod
    def load(cls, model_path: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_path: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model_path))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround byhttps://github.com/highway11git
            # to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location="cpu")

        model = cls._init_model_with_state_dict(state)

        if "model_card" in state:
            model.model_card = state["model_card"]

        model.eval()
        model.to(flair.device)

        return model

    def print_model_card(self):
        if hasattr(self, "model_card"):
            param_out = "\n------------------------------------\n"
            param_out += "--------- Flair Model Card ---------\n"
            param_out += "------------------------------------\n"
            param_out += "- this Flair model was trained with:\n"
            param_out += f"-- Flair version {self.model_card['flair_version']}\n"
            param_out += f"-- PyTorch version {self.model_card['pytorch_version']}\n"
            if "transformers_version" in self.model_card:
                param_out += "-- Transformers version " f"{self.model_card['transformers_version']}\n"
            param_out += "------------------------------------\n"

            param_out += "------- Training Parameters: -------\n"
            param_out += "------------------------------------\n"
            training_params = "\n".join(
                f'-- {param} = {self.model_card["training_parameters"][param]}'
                for param in self.model_card["training_parameters"]
            )
            param_out += training_params + "\n"
            param_out += "------------------------------------\n"

            log.info(param_out)
        else:
            log.info(
                "This model has no model card (likely because it is not yet "
                "trained or was trained with Flair version < 0.9.1)"
            )


class Classifier(Model[DT], typing.Generic[DT]):
    """Abstract base class for all Flair models that do classification,
    both single- and multi-label. It inherits from flair.nn.Model and adds an
    unified evaluate() function so that all classification models use the same
    evaluation routines and compute the same numbers."""

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}

            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")
                # predict for batch
                loss_and_count = self.predict(
                    typing.cast(List[DT], batch),
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=return_loss,
                    embedding_storage_mode=embedding_storage_mode,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0].sum()
                    else:
                        eval_loss += loss_and_count.sum()

                # get the gold labels
                for datapoint in batch:

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    for predicted_span in datapoint.get_labels("predicted"):
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                        else:
                            all_predicted_values[representation].append(predicted_span.value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"]
                )

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ]
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]

        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

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
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        return result

    @abstractmethod
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
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.  # noqa: E501
        """
        raise NotImplementedError

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # check if there is a label mismatch
            g = [label.labeled_identifier for label in datapoint.get_labels(gold_label_type)]
            p = [label.labeled_identifier for label in datapoint.get_labels("predicted")]
            g.sort()
            p.sort()
            correct_string = " -> MISMATCH!\n" if g != p else ""
            # print info
            eval_line = (
                f"{datapoint.to_original_text()}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\n{correct_string}\n"
            )
            lines.append(eval_line)
        return lines


class DefaultClassifier(Classifier[DT], typing.Generic[DT, DT2]):
    """Default base class for all Flair models that do classification, both
    single- and multi-label. It inherits from flair.nn.Classifier and thus from
    flair.nn.Model. All features shared by all classifiers are implemented here,
    including the loss calculation and the predict() method. Currently, the
    TextClassifier, RelationExtractor, TextPairClassifier and
    SimpleSequenceTagger implement this class.
    """

    def __init__(
        self,
        label_dictionary: Dictionary,
        final_embedding_size: int,
        dropout: float = 0.0,
        locked_dropout: float = 0.0,
        word_dropout: float = 0.0,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        loss_weights: Dict[str, float] = None,
        decoder: Optional[torch.nn.Module] = None,
        inverse_model: bool = False,
        train_on_gold_pairs_only: bool = False,
    ):

        super().__init__()

        # initialize the label dictionary
        self.label_dictionary: Dictionary = label_dictionary

        # initialize the decoder
        if decoder is not None:
            self.decoder = decoder
            self._custom_decoder = True
        else:
            self.decoder = torch.nn.Linear(final_embedding_size, len(self.label_dictionary))
            torch.nn.init.xavier_uniform_(self.decoder.weight)
            self._custom_decoder = False

        # set up multi-label logic
        self.multi_label = multi_label
        self.multi_label_threshold = multi_label_threshold
        self.final_embedding_size = final_embedding_size
        self.inverse_model = inverse_model

        # init dropouts
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
        self.word_dropout = flair.nn.WordDropout(word_dropout)

        # loss weights and loss function
        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1.0 for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights: Optional[torch.Tensor] = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # set up gradient reversal if so specified
        if inverse_model:
            from pytorch_revgrad import RevGrad

            self.gradient_reversal = RevGrad()

        if self.multi_label:
            self.loss_function: _Loss = torch.nn.BCEWithLogitsLoss(weight=self.loss_weights, reduction="none")
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="none")
        self.train_on_gold_pairs_only = train_on_gold_pairs_only

    def _filter_data_point(self, data_point: DT) -> bool:
        """Specify if a data point should be kept. That way you can remove for example empty texts.
        Return true if the data point should be kept and false if it should be removed.
        """
        return True

    @abstractmethod
    def _embed_prediction_data_point(self, prediction_data_point: DT2) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def _inner_embeddings(self) -> Optional[Embeddings[DT]]:
        return None

    def _prepare_tensors(self, data_points: List[DT]) -> Tuple[torch.Tensor, ...]:
        filtered_data_points = [dt for dt in data_points if self._filter_data_point(dt)]
        if not filtered_data_points:
            return (torch.zeros(0, self.final_embedding_size, device=flair.device),)
        if self._inner_embeddings is not None:
            self._inner_embeddings.embed(filtered_data_points)
        embedding_list = []
        for prediction_data_point in self._get_prediction_data_points(filtered_data_points):
            embedding_list.append(self._embed_prediction_data_point(prediction_data_point).unsqueeze(0))

        embedded_data_pairs = torch.cat(embedding_list, 0)

        return (embedded_data_pairs,)

    def _transform_embeddings(
        self,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        """This method does applies a transformation through the model given a list of tensors as input.
        Returns the embeddings that will ran trough a linear decoder layer to calculate logits.

        If it is not overwritten, it will act as identity function for of the first tensor.
        """
        return args[0]

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        emb = self._transform_embeddings(*args)
        emb = emb.unsqueeze(1)
        emb = self.dropout(emb)
        emb = self.locked_dropout(emb)
        emb = self.word_dropout(emb)
        emb = emb.squeeze(1)

        if self.inverse_model:
            emb = self.gradient_reversal(emb)

        scores = self.decoder(emb)
        return scores

    @abstractmethod
    def _get_prediction_data_points(self, sentences: List[DT]) -> List[DT2]:
        """Returns the data_points to which labels are added (Sentence, Span, Token, ... objects)"""
        raise NotImplementedError

    def _get_label_of_datapoint(self, data_point: DT2) -> List[str]:
        """Extracts the labels from the data points.
        Each data point might return a list of strings, representing multiple labels.
        """
        if self.multi_label:
            return [label.value for label in data_point.get_labels(self.label_type)]
        else:
            return [data_point.get_label(self.label_type).value]

    @property
    def multi_label_threshold(self):
        return self._multi_label_threshold

    @multi_label_threshold.setter
    def multi_label_threshold(self, x):  # setter method
        if type(x) is dict:
            if "default" in x:
                self._multi_label_threshold = x
            else:
                raise Exception('multi_label_threshold dict should have a "default" key')
        else:
            self._multi_label_threshold = {"default": x}

    def get_scores_and_labels(self, batch: List[DT]) -> Tuple[torch.Tensor, List[List[str]]]:
        batch = [dp for dp in batch if self._filter_data_point(dp)]
        predict_data_points = self._get_prediction_data_points(batch)
        labels = [self._get_label_of_datapoint(pdp) for pdp in predict_data_points]
        embedded_tensor = self._prepare_tensors(batch)
        logits = self._transform_embeddings(*embedded_tensor)
        return logits, labels

    def _prepare_label_tensor(self, prediction_data_points: List[DT2]) -> torch.Tensor:
        labels = [self._get_label_of_datapoint(dp) for dp in prediction_data_points]
        if self.multi_label:
            return torch.tensor(
                [
                    [1 if label in all_labels_for_point else 0 for label in self.label_dictionary.get_items()]
                    for all_labels_for_point in labels
                ],
                dtype=torch.float,
                device=flair.device,
            )
        else:
            return torch.tensor(
                [
                    self.label_dictionary.get_idx_for_item(label[0])
                    if len(label) > 0
                    else self.label_dictionary.get_idx_for_item("O")
                    for label in labels
                ],
                dtype=torch.long,
                device=flair.device,
            )

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:

        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]
        predict_data_points = self._get_prediction_data_points(sentences)
        labels = self._prepare_label_tensor(predict_data_points)

        # no loss can be calculated if there are no labels
        if labels.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        embedded_tensor = self._prepare_tensors(sentences)
        scores = self.forward(*embedded_tensor)

        # calculate the loss
        return self._calculate_loss(scores, labels)

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        losses = self.loss_function(scores, labels)
        return losses, labels.size(0)

    def _sort_data(self, data_points: List[DT]) -> List[DT]:

        if len(data_points) == 0:
            return []

        if not isinstance(data_points[0], Sentence):
            return data_points

        # filter empty sentences
        sentences = [sentence for sentence in typing.cast(List[Sentence], data_points) if len(sentence) > 0]

        # reverse sort all sequences by their length
        reordered_sentences = sorted(sentences, key=len, reverse=True)

        return typing.cast(List[DT], reordered_sentences)

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
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.  # noqa: E501
        'gpu' to store embeddings in GPU memory.
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

                batch = [dp for dp in batch if self._filter_data_point(dp)]

                # stop if all sentences are empty
                if not batch:
                    continue

                data_points = self._get_prediction_data_points(batch)

                if not data_points:
                    continue

                tensors = self._prepare_tensors(batch)

                scores = self.forward(*tensors)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    for sentence in data_points:
                        sentence.remove_labels(label_name)

                    if return_loss:
                        gold_labels = self._prepare_label_tensor(data_points)
                        losses = self._calculate_loss(scores, gold_labels)[0]
                        overall_loss += losses.sum()
                        label_count += len(data_points)

                    if self.multi_label:
                        sigmoided = torch.sigmoid(scores)  # size: (n_sentences, n_classes)
                        n_labels = sigmoided.size(1)
                        for s_idx, data_point in enumerate(data_points):
                            for l_idx in range(n_labels):
                                label_value = self.label_dictionary.get_item_for_index(l_idx)
                                if label_value == "O":
                                    continue
                                label_threshold = self._get_label_threshold(label_value)
                                label_score = sigmoided[s_idx, l_idx].item()
                                if label_score > label_threshold or return_probabilities_for_all_classes:
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                    else:
                        softmax = torch.nn.functional.softmax(scores, dim=-1)

                        if return_probabilities_for_all_classes:
                            n_labels = softmax.size(1)
                            for s_idx, data_point in enumerate(data_points):
                                for l_idx in range(n_labels):
                                    label_value = self.label_dictionary.get_item_for_index(l_idx)
                                    if label_value == "O":
                                        continue
                                    label_score = softmax[s_idx, l_idx].item()
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                        else:
                            conf, idx = torch.max(softmax, dim=-1)
                            for data_point, c, i in zip(data_points, conf, idx):
                                label_value = self.label_dictionary.get_item_for_index(i.item())
                                if label_value == "O":
                                    continue
                                data_point.add_label(typename=label_name, value=label_value, score=c.item())

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _get_label_threshold(self, label_value):
        label_threshold = self.multi_label_threshold["default"]
        if label_value in self.multi_label_threshold:
            label_threshold = self.multi_label_threshold[label_value]

        return label_threshold

    def __str__(self):
        return (
            super(flair.nn.Model, self).__str__().rstrip(")")
            + f"  (weights): {self.weight_dict}\n"
            + f"  (weight_tensor) {self.loss_weights}\n)"
        )

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        # add DefaultClassifier arguments
        for arg in [
            "decoder",
            "dropout",
            "word_dropout",
            "locked_dropout",
            "multi_label",
            "multi_label_threshold",
            "loss_weights",
            "train_on_gold_pairs_only",
            "inverse_model",
        ]:
            if arg not in kwargs and arg in state:
                kwargs[arg] = state[arg]

        return super(Classifier, cls)._init_model_with_state_dict(state, **kwargs)

    def _get_state_dict(self):
        state = super()._get_state_dict()

        # add variables of DefaultClassifier
        state["dropout"] = self.dropout.p
        state["word_dropout"] = self.word_dropout.dropout_rate
        state["locked_dropout"] = self.locked_dropout.dropout_rate
        state["multi_label"] = self.multi_label
        state["multi_label_threshold"] = self.multi_label_threshold
        state["loss_weights"] = self.loss_weights
        state["train_on_gold_pairs_only"] = self.train_on_gold_pairs_only
        state["inverse_model"] = self.inverse_model
        if self._custom_decoder:
            state["decoder"] = self.decoder

        return state
