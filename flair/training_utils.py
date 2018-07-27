from typing import List

import os
import numpy as np

from flair.data import Dictionary, Sentence


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
            return self._tp / (self._tp + self._fp)
        return 0.0

    def recall(self):
        if self._tp + self._fn > 0:
            return self._tp / (self._tp + self._fn)
        return 0.0

    def f_score(self):
        if self.precision() + self.recall() > 0:
            return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return 0.0

    def accuracy(self):
        if self._tp + self._tn + self._fp + self._fn > 0:
            return (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
        return 0.0

    def __str__(self):
        return '{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
            self.name, self.precision(), self.recall(), self.accuracy(), self.f_score())

    def print(self):
        print('{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
                self.name, self.precision(), self.recall(), self.accuracy(), self.f_score()))


def clear_embeddings(sentences: List[Sentence]):
    """
    Clears the embeddings from all given sentences.
    :param sentences: list of sentences
    """
    for sentence in sentences:
        for token in sentence.tokens:
            token.clear_embeddings()


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
    converted_label_list = []

    for labels in label_list:
        arr = np.empty(len(label_dict))
        arr.fill(0)

        for label in labels:
            arr[label_dict.get_idx_for_item(label)] = 1

        converted_label_list.append(arr.tolist())

    return converted_label_list


def calculate_overall_metric(y_true: List[List[int]], y_pred: List[List[int]], labels: Dictionary) -> Metric:
    """
    Calculates the overall metrics (micro averaged) for the given predictions.
    The labels should be converted into a one-hot-list.
    :param y_true: list of true labels
    :param y_pred: list of predicted labels
    :param labels: the label dictionary
    :return: the overall metrics
    """
    metric = Metric("OVERALL")

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