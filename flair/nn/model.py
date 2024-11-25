import inspect
import itertools
import logging
import typing
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import torch.nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair
from flair.class_utils import get_non_abstract_subclasses
from flair.data import DT, DT2, Corpus, Dictionary, Sentence, _iter_dataset
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.distributed_utils import is_main_process
from flair.embeddings import Embeddings
from flair.embeddings.base import load_embeddings
from flair.file_utils import Tqdm, load_torch_state
from flair.training_utils import EmbeddingStorageMode, Result, store_embeddings

log = logging.getLogger("flair")


class Model(torch.nn.Module, typing.Generic[DT], ABC):
    """Abstract base class for all downstream task models in Flair, such as SequenceTagger and TextClassifier.

    Every new type of model must implement these methods.
    """

    model_card: Optional[dict[str, Any]] = None

    @property
    @abstractmethod
    def label_type(self) -> str:
        """Each model predicts labels of a certain type."""
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, data_points: list[DT]) -> tuple[torch.Tensor, int]:
        """Performs a forward pass and returns a loss tensor for backpropagation.

        Implement this to enable training.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data_points: Union[list[DT], Dataset],
        gold_label_type: str,
        out_path: Optional[Union[str, Path]] = None,
        embedding_storage_mode: EmbeddingStorageMode = "none",
        mini_batch_size: int = 32,
        main_evaluation_metric: tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: Optional[list[str]] = None,
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation results and a loss value.

        Implement this to enable evaluation.

        Args:
            data_points: The labeled data_points to evaluate.
            gold_label_type: The label type indicating the gold labels
            out_path: Optional output path to store predictions
            embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and freshly
              recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
            mini_batch_size: The batch_size to use for predictions
            main_evaluation_metric: Specify which metric to highlight as main_score
            exclude_labels: Specify classes that won't be considered in evaluation
            gold_label_dictionary: Specify which classes should be considered, all other classes will be taken as <unk>.
            return_loss: Weather to additionally compute the loss on the data-points.
            **kwargs: Arguments that will be ignored.

        Returns:
            The evaluation results.
        """
        exclude_labels = exclude_labels if exclude_labels is not None else []
        raise NotImplementedError

    def _get_state_dict(self) -> dict:
        """Returns the state dictionary for this model."""
        # Always include the name of the Model class for which the state dict holds
        state_dict = {"state_dict": self.state_dict(), "__cls__": self.__class__.__name__}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state: dict[str, Any], **kwargs):
        """Initialize the model from a state dictionary."""
        if "embeddings" in kwargs:
            embeddings = kwargs.pop("embeddings")
            if isinstance(embeddings, dict):
                embeddings = load_embeddings(embeddings)
            kwargs["embeddings"] = embeddings

        model = cls(**kwargs)

        model.load_state_dict(state["state_dict"])

        return model

    @staticmethod
    def _fetch_model(model_name):
        # this seems to just return model name, not a model with that name
        return model_name

    def save(self, model_file: Union[str, Path], checkpoint: bool = False) -> None:
        """Saves the current model to the provided file.

        Args:
            model_file: the model file
            checkpoint: currently unused.
        """
        model_state = self._get_state_dict()

        # write out a "model card" if one is set
        if self.model_card is not None:
            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "Model":
        """Loads the model from the given file.

        Args:
            model_path: the model file or the already loaded state dict

        Returns: the loaded text classifier model
        """
        # if this class is abstract, go through all inheriting classes and try to fetch and load the model
        if inspect.isabstract(cls):
            # get all non-abstract subclasses
            subclasses = list(get_non_abstract_subclasses(cls))

            # try to fetch the model for each subclass. if fetching is possible, load model and return it
            for model_cls in subclasses:
                try:
                    new_model_path = model_cls._fetch_model(model_path)
                    if new_model_path != model_path:
                        return model_cls.load(new_model_path)
                except Exception as e:
                    log.debug(e)
                    # skip any invalid loadings, e.g. not found on HuggingFace hub
                    continue

            # if the model cannot be fetched, load as a file
            try:
                state = model_path if isinstance(model_path, dict) else load_torch_state(str(model_path))
            except Exception:
                log.error("-" * 80)
                log.error(
                    f"ERROR: The key '{model_path}' was neither found on the ModelHub nor is this a valid path to a file on your system!"
                )
                log.error(" -> Please check https://huggingface.co/models?filter=flair for all available models.")
                log.error(" -> Alternatively, point to a model file on your local drive.")
                log.error("-" * 80)
                raise ValueError(f"Could not find any model with name '{model_path}'")

            # try to get model class from state
            cls_name = state.pop("__cls__", None)
            if cls_name:
                for model_cls in subclasses:
                    if cls_name == model_cls.__name__:
                        return model_cls.load(state)

            # older (flair 11.3 and below) models do not contain cls information. In this case, try all subclasses
            for model_cls in subclasses:
                try:
                    model = model_cls.load(state)
                    return model
                except Exception as e:
                    print(e)
                    # skip any invalid loadings, e.g. not found on HuggingFace hub
                    continue

            raise ValueError(f"Could not find any model with name '{model_path}'")

        else:
            # if this class is not abstract, fetch the model and load it
            if not isinstance(model_path, dict):
                model_file = cls._fetch_model(str(model_path))
                state = load_torch_state(model_file)
            else:
                state = model_path

            if "__cls__" in state:
                state.pop("__cls__")

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
                param_out += f"-- Transformers version {self.model_card['transformers_version']}\n"
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


class ReduceTransformerVocabMixin(ABC):
    @abstractmethod
    def get_used_tokens(
        self, corpus: Corpus, context_lenth: int = 0, respect_document_boundaries: bool = True
    ) -> typing.Iterable[list[str]]:
        pass


class Classifier(Model[DT], typing.Generic[DT], ReduceTransformerVocabMixin, ABC):
    """Abstract base class for all Flair models that do classification.

    The classifier inherits from flair.nn.Model and adds unified functionality for both, single- and multi-label
    classification and evaluation. Therefore, it is ensured to have a fair comparison between multiple classifiers.
    """

    def evaluate(
        self,
        data_points: Union[list[DT], Dataset],
        gold_label_type: str,
        out_path: Optional[Union[str, Path]] = None,
        embedding_storage_mode: EmbeddingStorageMode = "none",
        mini_batch_size: int = 32,
        main_evaluation_metric: tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: Optional[list[str]] = None,
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        exclude_labels = exclude_labels if exclude_labels is not None else []

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
            lines: list[str] = []

            # variables for computing scores
            all_spans: set[str] = set()
            all_true_values = {}
            all_predicted_values = {}

            loader = DataLoader(data_points, batch_size=mini_batch_size)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader, disable=not is_main_process()):
                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=return_loss,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count

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
                list_of_gold_values_for_span = all_true_values.get(span, ["O"])
                # delete excluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(all_predicted_values.get(span, ["O"]))

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

        log.debug(f"Evaluating as a multi-label problem: {multi_label}")

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

        for label_name, _count in counter.most_common():
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

            # compute accuracy separately as it is not always in classification_report (e.g. when micro avg exists)
            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            # The "micro avg" appears only in the classification report if no prediction is possible.
            # Otherwise, it is identical to the "macro avg". In this case, we add it to the report.
            if "micro avg" not in classification_report_dict:
                classification_report_dict["micro avg"] = {}
                for metric_key in classification_report_dict["macro avg"]:
                    if metric_key != "support":
                        classification_report_dict["micro avg"][metric_key] = classification_report_dict["accuracy"]
                    else:
                        classification_report_dict["micro avg"][metric_key] = classification_report_dict["macro avg"][
                            "support"
                        ]

            detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {round(classification_report_dict['micro avg']['f1-score'], 4)}"
                f"\n- F-score (macro) {round(classification_report_dict['macro avg']['f1-score'], 4)}"
                f"\n- Accuracy {accuracy_score}"
                "\n\nBy class:\n" + classification_report
            )

            # Create and populate score object for logging with all evaluation values, plus the loss
            scores: dict[Union[tuple[str, ...], str], Any] = {}

            for avg_type in ("micro avg", "macro avg"):
                for metric_type in ("f1-score", "precision", "recall"):
                    scores[(avg_type, metric_type)] = classification_report_dict[avg_type][metric_type]

            scores["accuracy"] = accuracy_score

            if average_over > 0:
                eval_loss /= average_over
            scores["loss"] = eval_loss.item()

            return Result(
                main_score=classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]],
                detailed_results=detailed_result,
                classification_report=classification_report_dict,
                scores=scores,
            )

        else:
            # issue error and default all evaluation numbers to 0.
            error_text = (
                f"It was not possible to compute evaluation values because: \n"
                f"- The evaluation data has no gold labels for label_type='{gold_label_type}'!\n"
                f"- And no predictions were made!\n"
                "Double check your corpus (if the test split has labels), and how you initialize the ModelTrainer!"
            )

            return Result(
                main_score=0.0,
                detailed_results=error_text,
                classification_report={},
                scores={"loss": 0.0},
            )

    @abstractmethod
    def predict(
        self,
        sentences: Union[list[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: EmbeddingStorageMode = "none",
    ):
        """Predicts the class labels for the given sentences.

        The labels are directly added to the sentences.

        Args:
            sentences: list of sentences
            mini_batch_size: mini batch size to use
            return_probabilities_for_all_classes: return probabilities for all classes instead of only best predicted
            verbose: set to True to display a progress bar
            return_loss: set to True to return loss
            label_name: set this to change the name of the label type that is predicted  # noqa: E501
            embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.  # noqa: E501
        """
        raise NotImplementedError

    def _print_predictions(self, batch: list[DT], gold_label_type: str) -> list[str]:
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
                f"{datapoint.text}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\n{correct_string}\n"
            )
            lines.append(eval_line)
        return lines

    def get_used_tokens(
        self, corpus: Corpus, context_length: int = 0, respect_document_boundaries: bool = True
    ) -> typing.Iterable[list[str]]:
        for sentence in _iter_dataset(corpus.get_all_sentences()):
            yield [t.text for t in sentence]
            yield [t.text for t in sentence.left_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence.right_context(context_length, respect_document_boundaries)]

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "Classifier":
        from typing import cast

        return cast("Classifier", super().load(model_path=model_path))


class DefaultClassifier(Classifier[DT], typing.Generic[DT, DT2], ABC):
    """Default base class for all Flair models that do classification.

    It inherits from flair.nn.Classifier and thus from flair.nn.Model. All features shared by all classifiers are
    implemented here, including the loss calculation, prediction heads for both single- and multi- label classification
    and the `predict()` method. Example implementations of this class are the TextClassifier, RelationExtractor,
    TextPairClassifier and TokenClassifier.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        label_dictionary: Dictionary,
        final_embedding_size: int,
        dropout: float = 0.0,
        locked_dropout: float = 0.0,
        word_dropout: float = 0.0,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        loss_weights: Optional[dict[str, float]] = None,
        decoder: Optional[torch.nn.Module] = None,
        inverse_model: bool = False,
        train_on_gold_pairs_only: bool = False,
        should_embed_sentence: bool = True,
    ) -> None:
        super().__init__()

        # set the embeddings
        self.embeddings = embeddings

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
        self.should_embed_sentence = should_embed_sentence

        # loss weights and loss function
        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1.0 for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights:
                    weight_list[i] = loss_weights[tag]
            self.loss_weights: Optional[torch.Tensor] = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # set up gradient reversal if so specified
        if inverse_model:
            from pytorch_revgrad import RevGrad

            self.gradient_reversal = RevGrad()

        if self.multi_label:
            self.loss_function: _Loss = torch.nn.BCEWithLogitsLoss(weight=self.loss_weights, reduction="sum")
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")
        self.train_on_gold_pairs_only = train_on_gold_pairs_only

    def _filter_data_point(self, data_point: DT) -> bool:
        """Specify if a data point should be kept.

        That way you can remove for example empty texts. Per default all datapoints that have length zero
        will be removed.
        Return true if the data point should be kept and false if it should be removed.
        """
        return len(data_point) > 0

    @abstractmethod
    def _get_embedding_for_data_point(self, prediction_data_point: DT2) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_data_points_from_sentence(self, sentence: DT) -> list[DT2]:
        """Returns the data_points to which labels are added.

        The results should be of any type that inherits from DataPoint (Sentence, Span, Token, ... objects).
        """
        raise NotImplementedError

    def _get_data_points_for_batch(self, sentences: list[DT]) -> list[DT2]:
        """Returns the data_points to which labels are added.

        The results should be of any type that inherits from DataPoint (Sentence, Span, Token, ... objects).
        """
        return [data_point for sentence in sentences for data_point in self._get_data_points_from_sentence(sentence)]

    def _get_label_of_datapoint(self, data_point: DT2) -> list[str]:
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
        if isinstance(x, dict):
            if "default" in x:
                self._multi_label_threshold = x
            else:
                raise ValueError('multi_label_threshold dict should have a "default" key')
        else:
            self._multi_label_threshold = {"default": x}

    def _prepare_label_tensor(self, prediction_data_points: list[DT2]) -> torch.Tensor:
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
                    (
                        self.label_dictionary.get_idx_for_item(label[0])
                        if len(label) > 0
                        else self.label_dictionary.get_idx_for_item("O")
                    )
                    for label in labels
                ],
                dtype=torch.long,
                device=flair.device,
            )

    def _encode_data_points(self, sentences: list[DT], data_points: list[DT2]) -> Tensor:
        # embed sentences
        if self.should_embed_sentence:
            self.embeddings.embed(sentences)

        # get a tensor of data points
        data_point_tensor = torch.stack([self._get_embedding_for_data_point(data_point) for data_point in data_points])

        # do dropout
        data_point_tensor = data_point_tensor.unsqueeze(1)
        data_point_tensor = self.dropout(data_point_tensor)
        data_point_tensor = self.locked_dropout(data_point_tensor)
        data_point_tensor = self.word_dropout(data_point_tensor)
        data_point_tensor = data_point_tensor.squeeze(1)

        return data_point_tensor

    def _mask_scores(self, scores: Tensor, data_points) -> Tensor:
        """Classes that inherit from DefaultClassifier may optionally mask scores."""
        return scores

    def forward_loss(self, sentences: list[DT]) -> tuple[torch.Tensor, int]:
        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        label_tensor = self._prepare_label_tensor(data_points)
        if label_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # decode
        scores = self.decoder(data_point_tensor)

        # an optional masking step (no masking in most cases)
        scores = self._mask_scores(scores, data_points)

        # calculate the loss
        return self._calculate_loss(scores, label_tensor)

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.loss_function(scores, labels), labels.size(0)

    def _sort_data(self, data_points: list[DT]) -> list[DT]:
        if len(data_points) == 0:
            return []

        if not isinstance(data_points[0], Sentence):
            return data_points

        # filter empty sentences
        sentences = [sentence for sentence in typing.cast(list[Sentence], data_points) if len(sentence) > 0]

        # reverse sort all sequences by their length
        reordered_sentences = sorted(sentences, key=len, reverse=True)

        return typing.cast(list[DT], reordered_sentences)

    def predict(
        self,
        sentences: Union[list[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: EmbeddingStorageMode = "none",
    ):
        """Predicts the class labels for the given sentences. The labels are directly added to the sentences.

        Args:
            sentences: list of sentences to predict
            mini_batch_size: the amount of sentences that will be predicted within one batch
            return_probabilities_for_all_classes: return probabilities for all classes instead of only best predicted
            verbose: set to True to display a progress bar
            return_loss: set to True to return loss
            label_name: set this to change the name of the label type that is predicted
            embedding_storage_mode: default is 'none' which is the best is most cases.
                Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            if isinstance(sentences[0], Sentence):
                Sentence.set_context_for_sentences(typing.cast(list[Sentence], sentences))

            reordered_sentences = self._sort_data(sentences)

            if len(reordered_sentences) == 0:
                return sentences

            if len(reordered_sentences) > mini_batch_size:
                batches: Union[DataLoader, list[list[DT]]] = DataLoader(
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
            has_any_unknown_label = False
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
                scores = self._mask_scores(scores, data_points)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    for sentence in data_points:
                        sentence.remove_labels(label_name)

                    if return_loss:
                        # filter data points that have labels outside of dictionary
                        filtered_indices = []
                        has_unknown_label = False
                        for idx, dp in enumerate(data_points):
                            if all(
                                label in self.label_dictionary.get_items() for label in self._get_label_of_datapoint(dp)
                            ):
                                filtered_indices.append(idx)
                            else:
                                has_unknown_label = True

                        if has_unknown_label:
                            has_any_unknown_label = True
                            scores = torch.index_select(scores, 0, torch.tensor(filtered_indices, device=flair.device))

                        gold_labels = self._prepare_label_tensor([data_points[index] for index in filtered_indices])
                        overall_loss += self._calculate_loss(scores, gold_labels)[0]
                        label_count += len(filtered_indices)

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
                            conf, indices = torch.max(softmax, dim=-1)
                            for data_point, c, i in zip(data_points, conf, indices):
                                label_value = self.label_dictionary.get_item_for_index(i.item())
                                if label_value == "O":
                                    continue
                                data_point.add_label(typename=label_name, value=label_value, score=c.item())

                store_embeddings(batch, storage_mode=embedding_storage_mode)

                self._post_process_batch_after_prediction(batch, label_name)

            if return_loss:
                if has_any_unknown_label:
                    log.info(
                        "During evaluation, encountered labels that are not in the label_dictionary:"
                        "Evaluation loss is computed without them."
                    )
                return overall_loss, label_count
            return None

    def _post_process_batch_after_prediction(self, batch, label_name):
        pass

    def _get_label_threshold(self, label_value):
        label_threshold = self.multi_label_threshold["default"]
        if label_value in self.multi_label_threshold:
            label_threshold = self.multi_label_threshold[label_value]

        return label_threshold

    def __str__(self) -> str:
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

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "DefaultClassifier":
        from typing import cast

        return cast("DefaultClassifier", super().load(model_path=model_path))
