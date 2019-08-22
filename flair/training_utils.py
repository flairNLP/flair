import itertools
import random
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from flair.data import Dictionary, Sentence
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from abc import abstractmethod


class Result(object):
    def __init__(
        self, main_score: float, log_header: str, log_line: str, detailed_results: str
    ):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


class Metric(object):
    def __init__(self, name):
        self.name = name

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)
            > 0
        ):
            return round(
                (self.get_tp(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                ),
                4,
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


class MetricRegression(object):
    def __init__(self, name):
        self.name = name

        self.true = []
        self.pred = []

    def mean_squared_error(self):
        return mean_squared_error(self.true, self.pred)

    def mean_absolute_error(self):
        return mean_absolute_error(self.true, self.pred)

    def pearsonr(self):
        return pearsonr(self.true, self.pred)[0]

    def spearmanr(self):
        return spearmanr(self.true, self.pred)[0]

    ## dummy return to fulfill trainer.train() needs
    def micro_avg_f_score(self):
        return self.mean_squared_error()

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_MEAN_SQUARED_ERROR\t{0}_MEAN_ABSOLUTE_ERROR\t{0}_PEARSON\t{0}_SPEARMAN".format(
                prefix
            )

        return "MEAN_SQUARED_ERROR\tMEAN_ABSOLUTE_ERROR\tPEARSON\tSPEARMAN"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        line = "mean squared error: {0:.4f} - mean absolute error: {1:.4f} - pearson: {2:.4f} - spearman: {3:.4f}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )
        return line


class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"


class WeightExtractor(object):
    def __init__(self, directory: Path, number_of_weights: int = 10):
        self.weights_file = init_output_file(directory, "weights.txt")
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            weights_to_watch = min(
                self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size()))
            )

            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]
                for index in self.weights_dict[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(self.weights_file, "a") as f:
                    f.write("{}\t{}\t{}\t{}\n".format(iteration, key, i, float(value)))

    def _init_weights_index(self, key, state_dict, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = state_dict[key]
            cur_indices = []

            for x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        self.weights_dict[key] = indices


def init_output_file(base_path: Path, file_name: str) -> Path:
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, "w", encoding="utf-8").close()
    return file


def convert_labels_to_one_hot(
    label_list: List[List[str]], label_dict: Dictionary
) -> List[List[int]]:
    """
    Convert list of labels (strings) to a one hot list.
    :param label_list: list of labels
    :param label_dict: label dictionary
    :return: converted label list
    """
    return [
        [1 if l in labels else 0 for l in label_dict.get_items()]
        for labels in label_list
    ]


def log_line(log):
    log.info("-" * 100)


def add_file_handler(log, output_file):
    init_output_file(output_file.parents[0], output_file.name)
    fh = logging.FileHandler(output_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh


def store_embeddings(sentences: List[Sentence], storage_mode: str):

    # if memory mode option 'none' delete everything
    if storage_mode == "none":
        for sentence in sentences:
            sentence.clear_embeddings()

    # else delete only dynamic embeddings (otherwise autograd will keep everything in memory)
    else:
        # find out which ones are dynamic embeddings
        delete_keys = []
        for name, vector in sentences[0][0]._embeddings.items():
            if sentences[0][0]._embeddings[name].requires_grad:
                delete_keys.append(name)

        # find out which ones are dynamic embeddings
        for sentence in sentences:
            sentence.clear_embeddings(delete_keys)

    # memory management - option 1: send everything to CPU
    if storage_mode == "cpu":
        for sentence in sentences:
            sentence.to("cpu")
