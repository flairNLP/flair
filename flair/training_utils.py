import random
import logging
import os
from collections import defaultdict
from typing import List
from flair.data import Dictionary, Sentence
from functools import reduce


log = logging.getLogger(__name__)


class Metric(object):

    def __init__(self, name):
        self.name = name

        self._tp = 0.0
        self._fp = 0.0
        self._tn = 0.0
        self._fn = 0.0

    def tp(self):
        self._tp += 1

    def tn(self):
        self._tn += 1

    def fp(self):
        self._fp += 1

    def fn(self):
        self._fn += 1

    def precision(self):
        if self._tp + self._fp > 0:
            return round(self._tp / (self._tp + self._fp), 4)
        return 0.0

    def recall(self):
        if self._tp + self._fn > 0:
            return round(self._tp / (self._tp + self._fn), 4)
        return 0.0

    def f_score(self):
        if self.precision() + self.recall() > 0:
            return round(2 * (self.precision() * self.recall()) / (self.precision() + self.recall()), 4)
        return 0.0

    def accuracy(self):
        if self._tp + self._tn + self._fp + self._fn > 0:
            return round((self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn), 4)
        return 0.0

    def to_tsv(self):
        return '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            self._tp, self._tn, self._fp, self._fn, self.precision(), self.recall(), self.f_score(), self.accuracy())

    def print(self):
        log.info(self)

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return '{0}_TP\t{0}_TN\t{0}_FP\t{0}_FN\t{0}_PRECISION\t{0}_RECALL\t{0}_F-SCORE\t{0}_ACCURACY'.format(prefix)

        return 'TP\tTN\tFP\tFN\tPRECISION\tRECALL\tF-SCORE\tACCURACY'

    @staticmethod
    def to_empty_tsv():
        return '_\t_\t_\t_\t_\t_\t_\t_'

    def __str__(self):
        return '{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}'.format(
            self.name, self._tp, self._fp, self._fn, self._tn, self.precision(), self.recall(), self.accuracy(), self.f_score())


class WeightExtractor(object):

    def __init__(self, directory: str, number_of_weights: int = 10):
        self.weights_file = init_output_file(directory, 'weights.txt')
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            weights_to_watch = min(self.number_of_weights, reduce(lambda x, y: x*y, list(vec.size())))

            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]
                for index in self.weights_dict[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(self.weights_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\n'.format(iteration, key, i, float(value)))

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


def clear_embeddings(sentences: List[Sentence], also_clear_word_embeddings=False):
    """
    Clears the embeddings from all given sentences.
    :param sentences: list of sentences
    """
    for sentence in sentences:
        sentence.clear_embeddings(also_clear_word_embeddings=also_clear_word_embeddings)


def init_output_file(base_path: str, file_name: str):
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    os.makedirs(base_path, exist_ok=True)

    file = os.path.join(base_path, file_name)
    open(file, "w", encoding='utf-8').close()
    return file


def convert_labels_to_one_hot(label_list: List[List[str]], label_dict: Dictionary) -> List[List[int]]:
    """
    Convert list of labels (strings) to a one hot list.
    :param label_list: list of labels
    :param label_dict: label dictionary
    :return: converted label list
    """
    return [[1 if l in labels else 0 for l in label_dict.get_items()] for labels in label_list]


def calculate_micro_avg_metric(y_true: List[List[int]], y_pred: List[List[int]], labels: Dictionary) -> Metric:
    """
    Calculates the overall metrics (micro averaged) for the given predictions.
    The labels should be converted into a one-hot-list.
    :param y_true: list of true labels
    :param y_pred: list of predicted labels
    :param labels: the label dictionary
    :return: the overall metrics
    """
    metric = Metric("MICRO_AVG")

    for pred, true in zip(y_pred, y_true):
        for i in range(len(labels)):
            if true[i] == 1 and pred[i] == 1:
                metric.tp()
            elif true[i] == 1 and pred[i] == 0:
                metric.fn()
            elif true[i] == 0 and pred[i] == 1:
                metric.fp()
            elif true[i] == 0 and pred[i] == 0:
                metric.tn()

    return metric


def calculate_class_metrics(y_true: List[List[int]], y_pred: List[List[int]], labels: Dictionary) -> List[Metric]:
    """
    Calculates the metrics for the individual classes for the given predictions.
    The labels should be converted into a one-hot-list.
    :param y_true: list of true labels
    :param y_pred: list of predicted labels
    :param labels: the label dictionary
    :return: the metrics for every class
    """
    metrics = []

    for label in labels.get_items():
        metric = Metric(label)
        label_idx = labels.get_idx_for_item(label)

        for true, pred in zip(y_true, y_pred):
            if true[label_idx] == 1 and pred[label_idx] == 1:
                metric.tp()
            elif true[label_idx] == 1 and pred[label_idx] == 0:
                metric.fn()
            elif true[label_idx] == 0 and pred[label_idx] == 1:
                metric.fp()
            elif true[label_idx] == 0 and pred[label_idx] == 0:
                metric.tn()

        metrics.append(metric)

    return metrics
