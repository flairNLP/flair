from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import csv

import matplotlib
import math

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# header for 'weights.txt'
WEIGHT_NAME = 1
WEIGHT_NUMBER = 2
WEIGHT_VALUE = 3


class Plotter(object):
    """
    Plots training parameters (loss, f-score, and accuracy) and training weights over time.
    Input files are the output files 'loss.tsv' and 'weights.txt' from training either a sequence tagger or text
    classification model.
    """

    @staticmethod
    def _extract_evaluation_data(file_name: Path) -> dict:
        training_curves = {
            'train': {
                'loss': [],
                'f_score': [],
                'acc': []
            },
            'test': {
                'loss': [],
                'f_score': [],
                'acc': []
            },
            'dev': {
                'loss': [],
                'f_score': [],
                'acc': []
            }
        }

        with open(file_name, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')

            # determine the column index of loss, f-score and accuracy for train, dev and test split
            row = next(tsvin, None)
            TRAIN_LOSS = row.index('TRAIN_LOSS')
            TRAIN_F_SCORE = row.index('TRAIN_F-SCORE')
            TRAIN_ACCURACY = row.index('TRAIN_ACCURACY')
            DEV_LOSS = row.index('DEV_LOSS')
            DEV_F_SCORE = row.index('DEV_F-SCORE')
            DEV_ACCURACY = row.index('DEV_ACCURACY')
            TEST_LOSS = row.index('TEST_LOSS')
            TEST_F_SCORE = row.index('TEST_F-SCORE')
            TEST_ACCURACY = row.index('TEST_ACCURACY')

            # then get all relevant values from the tsv
            for row in tsvin:
                if row[TRAIN_LOSS] != '_': training_curves['train']['loss'].append(float(row[TRAIN_LOSS]))
                if row[TRAIN_F_SCORE] != '_': training_curves['train']['f_score'].append(float(row[TRAIN_F_SCORE]))
                if row[TRAIN_ACCURACY] != '_': training_curves['train']['acc'].append(float(row[TRAIN_ACCURACY]))
                if row[DEV_LOSS] != '_': training_curves['dev']['loss'].append(float(row[DEV_LOSS]))
                if row[DEV_F_SCORE] != '_': training_curves['dev']['f_score'].append(float(row[DEV_F_SCORE]))
                if row[DEV_ACCURACY] != '_': training_curves['dev']['acc'].append(float(row[DEV_ACCURACY]))
                if row[TEST_LOSS] != '_': training_curves['test']['loss'].append(float(row[TEST_LOSS]))
                if row[TEST_F_SCORE] != '_': training_curves['test']['f_score'].append(float(row[TEST_F_SCORE]))
                if row[TEST_ACCURACY] != '_': training_curves['test']['acc'].append(float(row[TEST_ACCURACY]))

        return training_curves

    @staticmethod
    def _extract_weight_data(file_name: Path) -> dict:
        weights = defaultdict(lambda: defaultdict(lambda: list()))

        with open(file_name, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')

            for row in tsvin:
                name = row[WEIGHT_NAME]
                param = row[WEIGHT_NUMBER]
                value = float(row[WEIGHT_VALUE])

                weights[name][param].append(value)

        return weights

    @staticmethod
    def _extract_learning_rate(file_name: Path):
        lrs = []
        losses = []

        with open(file_name, 'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            row = next(tsvin, None)
            LEARNING_RATE = row.index('LEARNING_RATE')
            TRAIN_LOSS = row.index('TRAIN_LOSS')

            # then get all relevant values from the tsv
            for row in tsvin:
                if row[TRAIN_LOSS] != '_': losses.append(float(row[TRAIN_LOSS]))
                if row[LEARNING_RATE] != '_': lrs.append(float(row[LEARNING_RATE]))

        return lrs, losses

    def plot_weights(self, file_name: Union[str, Path]):
        if type(file_name) is str:
            file_name = Path(file_name)

        weights = self._extract_weight_data(file_name)

        total = len(weights)
        columns = 2
        rows = max(2, int(math.ceil(total / columns)))

        figsize = (5, 5)
        if rows != columns:
            figsize = (5, rows + 5)

        fig = plt.figure()
        f, axarr = plt.subplots(rows, columns, figsize=figsize)

        c = 0
        r = 0
        for name, values in weights.items():
            # plot i
            axarr[r, c].set_title(name, fontsize=6)
            for _, v in values.items():
                axarr[r, c].plot(np.arange(0, len(v)), v, linewidth=0.35)
            axarr[r, c].set_yticks([])
            axarr[r, c].set_xticks([])
            c += 1
            if c == columns:
                c = 0
                r += 1

        while r != rows and c != columns:
            axarr[r, c].set_yticks([])
            axarr[r, c].set_xticks([])
            c += 1
            if c == columns:
                c = 0
                r += 1

        # save plots
        f.subplots_adjust(hspace=0.5)
        plt.tight_layout(pad=1.0)
        path = file_name.parent / 'weights.png'
        plt.savefig(path, dpi=300)

        plt.close(fig)

    def plot_training_curves(self, file_name: Union[str, Path]):
        if type(file_name) is str:
            file_name = Path(file_name)

        fig = plt.figure(figsize=(15, 10))

        training_curves = self._extract_evaluation_data(file_name)

        # plot 1
        plt.subplot(3, 1, 1)
        if training_curves['train']['loss']:
            x = np.arange(0, len(training_curves['train']['loss']))
            plt.plot(x, training_curves['train']['loss'], label='training loss')
        if training_curves['dev']['loss']:
            x = np.arange(0, len(training_curves['dev']['loss']))
            plt.plot(x, training_curves['dev']['loss'], label='validation loss')
        if training_curves['test']['loss']:
            x = np.arange(0, len(training_curves['test']['loss']))
            plt.plot(x, training_curves['test']['loss'], label='test loss')
        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.ylabel('loss')
        plt.xlabel('epochs')

        # plot 2
        plt.subplot(3, 1, 2)
        if training_curves['train']['acc']:
            x = np.arange(0, len(training_curves['train']['acc']))
            plt.plot(x, training_curves['train']['acc'], label='training accuracy')
        if training_curves['dev']['acc']:
            x = np.arange(0, len(training_curves['dev']['acc']))
            plt.plot(x, training_curves['dev']['acc'], label='validation accuracy')
        if training_curves['test']['acc']:
            x = np.arange(0, len(training_curves['test']['acc']))
            plt.plot(x, training_curves['test']['acc'], label='test accuracy')
        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.ylabel('accuracy')
        plt.xlabel('epochs')

        # plot 3
        plt.subplot(3, 1, 3)
        if training_curves['train']['f_score']:
            x = np.arange(0, len(training_curves['train']['f_score']))
            plt.plot(x, training_curves['train']['f_score'], label='training f1-score')
        if training_curves['dev']['f_score']:
            x = np.arange(0, len(training_curves['dev']['f_score']))
            plt.plot(x, training_curves['dev']['f_score'], label='validation f1-score')
        if training_curves['test']['f_score']:
            x = np.arange(0, len(training_curves['test']['f_score']))
            plt.plot(x, training_curves['test']['f_score'], label='test f1-score')
        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.ylabel('f1-score')
        plt.xlabel('epochs')

        # save plots
        plt.tight_layout(pad=1.0)
        path = file_name.parent / 'training.png'
        plt.savefig(path, dpi=300)

        plt.close(fig)

    def plot_learning_rate(self,
                           file_name: Union[str, Path],
                           skip_first: int = 10,
                           skip_last: int = 5):
        if type(file_name) is str:
            file_name = Path(file_name)

        lrs, losses = self._extract_learning_rate(file_name)
        lrs = lrs[skip_first:-skip_last] if skip_last > 0 else lrs[skip_first:]
        losses = losses[skip_first:-skip_last] if skip_last > 0 else losses[skip_first:]

        fig, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

        # save plot
        plt.tight_layout(pad=1.0)
        path = file_name.parent / 'learning_rate.png'
        plt.savefig(path, dpi=300)

        plt.close(fig)
