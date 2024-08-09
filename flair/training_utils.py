import logging
import random
from collections import defaultdict
from enum import Enum
from functools import reduce
from math import inf
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import Dataset

import flair
from flair.data import DT, Dictionary, Sentence, _iter_dataset

EmbeddingStorageMode = Literal["none", "cpu", "gpu"]
log = logging.getLogger("flair")


class Result:
    def __init__(
        self,
        main_score: float,
        detailed_results: str,
        classification_report: Optional[dict] = None,
        scores: Optional[Dict] = None,
    ) -> None:
        classification_report = classification_report if classification_report is not None else {}
        assert scores is not None and "loss" in scores, "No loss provided."

        self.main_score: float = main_score
        self.scores = scores
        self.detailed_results: str = detailed_results
        self.classification_report = classification_report

    @property
    def loss(self):
        return self.scores["loss"]

    def __str__(self) -> str:
        return f"{self.detailed_results!s}\nLoss: {self.loss}'"


class MetricRegression:
    def __init__(self, name) -> None:
        self.name = name

        self.true: List[float] = []
        self.pred: List[float] = []

    def mean_squared_error(self):
        return mean_squared_error(self.true, self.pred)

    def mean_absolute_error(self):
        return mean_absolute_error(self.true, self.pred)

    def pearsonr(self):
        return pearsonr(self.true, self.pred)[0]

    def spearmanr(self):
        return spearmanr(self.true, self.pred)[0]

    # dummy return to fulfill trainer.train() needs
    def micro_avg_f_score(self):
        return self.mean_squared_error()

    def to_tsv(self):
        return f"{self.mean_squared_error()}\t{self.mean_absolute_error()}\t{self.pearsonr()}\t{self.spearmanr()}"

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return f"{prefix}_MEAN_SQUARED_ERROR\t{prefix}_MEAN_ABSOLUTE_ERROR\t{prefix}_PEARSON\t{prefix}_SPEARMAN"

        return "MEAN_SQUARED_ERROR\tMEAN_ABSOLUTE_ERROR\tPEARSON\tSPEARMAN"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self) -> str:
        line = f"mean squared error: {self.mean_squared_error():.4f} - mean absolute error: {self.mean_absolute_error():.4f} - pearson: {self.pearsonr():.4f} - spearman: {self.spearmanr():.4f}"
        return line


class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"


class WeightExtractor:
    def __init__(self, directory: Union[str, Path], number_of_weights: int = 10) -> None:
        if isinstance(directory, str):
            directory = Path(directory)
        self.weights_file = init_output_file(directory, "weights.txt")
        self.weights_dict: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict:
            vec = state_dict[key]
            # print(vec)
            try:
                weights_to_watch = min(self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size())))
            except Exception:
                continue

            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]
                for index in self.weights_dict[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(self.weights_file, "a") as f:
                    f.write(f"{iteration}\t{key}\t{i}\t{float(value)}\n")

    def _init_weights_index(self, key, state_dict, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = state_dict[key]
            cur_indices = []

            for _x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        self.weights_dict[key] = indices


class AnnealOnPlateau:
    """A learningrate sheduler for annealing on plateau.

    This class is a modification of
    torch.optim.lr_scheduler.ReduceLROnPlateau that enables
    setting an "auxiliary metric" to break ties.
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
    ----
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
    -------
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        aux_mode="min",
        factor=0.1,
        patience=10,
        initial_extra_patience=0,
        verbose=False,
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ) -> None:
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.default_patience = patience
        self.effective_patience = patience + initial_extra_patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.aux_mode = aux_mode
        self.best = None
        self.best_aux = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metric, auxiliary_metric=None) -> bool:
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metric)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        is_better = False
        assert self.best is not None

        if self.mode == "min" and current < self.best:
            is_better = True

        if self.mode == "max" and current > self.best:
            is_better = True

        if current == self.best and auxiliary_metric:
            current_aux = float(auxiliary_metric)
            if self.aux_mode == "min" and current_aux < self.best_aux:
                is_better = True

            if self.aux_mode == "max" and current_aux > self.best_aux:
                is_better = True

        if is_better:
            self.best = current
            if auxiliary_metric:
                self.best_aux = auxiliary_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        reduce_learning_rate = self.num_bad_epochs > self.effective_patience
        if reduce_learning_rate:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.effective_patience = self.default_patience

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        return reduce_learning_rate

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    log.info(f" - reducing learning rate of group {epoch} to {new_lr}")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode)


def init_output_file(base_path: Union[str, Path], file_name: str) -> Path:
    """Creates a local file which can be appended to.

    Args:
        base_path: the path to the directory
        file_name: the file name

    Returns: the created file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    file.touch(exist_ok=True)
    return file


def convert_labels_to_one_hot(label_list: List[List[str]], label_dict: Dictionary) -> List[List[int]]:
    """Convert list of labels to a one hot list.

    Args:
        label_list: list of labels
        label_dict: label dictionary

    Returns: converted label list
    """
    return [[1 if label in labels else 0 for label in label_dict.get_items()] for labels in label_list]


def log_line(log):
    log.info("-" * 100, stacklevel=3)


def add_file_handler(log, output_file):
    init_output_file(output_file.parents[0], output_file.name)
    fh = logging.FileHandler(output_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh


def store_embeddings(
    data_points: Union[List[DT], Dataset],
    storage_mode: EmbeddingStorageMode,
    dynamic_embeddings: Optional[List[str]] = None,
):
    if isinstance(data_points, Dataset):
        data_points = list(_iter_dataset(data_points))

    # if memory mode option 'none' delete everything
    if storage_mode == "none":
        dynamic_embeddings = None

    # if dynamic embedding keys not passed, identify them automatically
    elif dynamic_embeddings is None:
        dynamic_embeddings = identify_dynamic_embeddings(data_points)

    # always delete dynamic embeddings
    for data_point in data_points:
        data_point.clear_embeddings(dynamic_embeddings)

    # if storage mode is "cpu", send everything to CPU (pin to memory if we train on GPU)
    if storage_mode == "cpu":
        pin_memory = str(flair.device) != "cpu"
        for data_point in data_points:
            data_point.to("cpu", pin_memory=pin_memory)


def identify_dynamic_embeddings(data_points: List[DT]) -> Optional[List]:
    dynamic_embeddings = []
    all_embeddings = []
    for data_point in data_points:
        if isinstance(data_point, Sentence):
            first_token = data_point[0]
            for name, vector in first_token._embeddings.items():
                if vector.requires_grad:
                    dynamic_embeddings.append(name)
                all_embeddings.append(name)

        for name, vector in data_point._embeddings.items():
            if vector.requires_grad:
                dynamic_embeddings.append(name)
            all_embeddings.append(name)
        if dynamic_embeddings:
            return dynamic_embeddings
    if not all_embeddings:
        return None
    return list(set(dynamic_embeddings))
