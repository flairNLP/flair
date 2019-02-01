import itertools
import random
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from flair.data import Dictionary, Sentence
from functools import reduce


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

    def get_tn(self, class_name):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fp(class_name)), 4)
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fn(class_name)), 4)
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(2 * (self.precision(class_name) * self.recall(class_name))
                         / (self.precision(class_name) + self.recall(class_name)), 4)
        return 0.0

    def accuracy(self, class_name=None):
        if self.get_tp(class_name) + self.get_tn(class_name) + self.get_fp(class_name) + self.get_fn(class_name) > 0:
            return round(
                (self.get_tp(class_name) + self.get_tn(class_name))
                / (self.get_tp(class_name) + self.get_tn(class_name) + self.get_fp(class_name) + self.get_fn(class_name)),
                4)
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_precisions = [self.precision(class_name) for class_name in self.get_classes()]
        class_recalls = [self.recall(class_name) for class_name in self.get_classes()]

        if len(class_precisions) > 0 and len(class_recalls) > 0:
            macro_precision = sum(class_precisions) / len(class_precisions)
            macro_recall = sum(class_recalls) / len(class_recalls)

            if macro_precision + macro_recall > 0:
                return round(2 * (macro_precision * macro_recall) / (macro_precision + macro_recall), 4)

        return 0.0

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [self.accuracy(class_name) for class_name in self.get_classes()]

        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(itertools.chain(*[list(keys) for keys
                                            in [self._tps.keys(), self._fps.keys(), self._tns.keys(),
                                                self._fns.keys()]]))
        all_classes = [class_name for class_name in all_classes if class_name is not None]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return '{}\t{}\t{}\t{}'.format(
            self.precision(),
            self.recall(),
            self.accuracy(),
            self.micro_avg_f_score(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return '{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE'.format(
                prefix)

        return 'PRECISION\tRECALL\tACCURACY\tF-SCORE'

    @staticmethod
    def to_empty_tsv():
        return '\t_\t_\t_\t_'

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            '{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}'.format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name), self.get_fp(class_name), self.get_fn(class_name), self.get_tn(class_name),
                self.precision(class_name), self.recall(class_name), self.accuracy(class_name),
                self.f_score(class_name))
            for class_name in all_classes]
        return '\n'.join(all_lines)


class EvaluationMetric(Enum):
    MICRO_ACCURACY = 'micro-average accuracy'
    MICRO_F1_SCORE = 'micro-average f1-score'
    MACRO_ACCURACY = 'macro-average accuracy'
    MACRO_F1_SCORE = 'macro-average f1-score'


class WeightExtractor(object):

    def __init__(self, directory: Path, number_of_weights: int = 10):
        self.weights_file = init_output_file(directory, 'weights.txt')
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            weights_to_watch = min(self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size())))

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


def init_output_file(base_path: Path, file_name: str) -> Path:
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
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


def log_line(log):
    log.info('-' * 100)


def add_file_handler(log, output_file):
    init_output_file(output_file.parents[0], output_file.name)
    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
