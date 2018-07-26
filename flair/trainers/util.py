from functools import reduce
import random
from typing import List

import numpy as np

from flair.data import Dictionary
from flair.trainers.metric import Metric


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