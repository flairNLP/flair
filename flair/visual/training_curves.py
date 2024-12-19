import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

# header for 'weights.txt'
WEIGHT_NAME = 1
WEIGHT_NUMBER = 2
WEIGHT_VALUE = 3

log = logging.getLogger("flair")


class Plotter:
    """Plots training parameters (loss, f-score, and accuracy) and training weights over time.

    Input files are the output files 'loss.tsv' and 'weights.txt' from
    training either a sequence tagger or text classification model.
    """

    @staticmethod
    def _extract_evaluation_data(file_name: Union[str, Path], score: str = "F1") -> dict:
        file_name = Path(file_name)

        training_curves: dict[str, dict[str, list[float]]] = {
            "train": {"loss": [], "score": []},
            "test": {"loss": [], "score": []},
            "dev": {"loss": [], "score": []},
        }

        with open(file_name) as f:
            tsvin = csv.reader(f, delimiter="\t")

            # determine the column index of loss, f-score and accuracy for
            # train, dev and test split
            row = next(tsvin)

            score = score.upper()

            if f"TEST_{score}" not in row:
                log.warning("-" * 100)
                log.warning(f"WARNING: No {score} found for test split in this data.")
                log.warning(f"Are you sure you want to plot {score} and not another value?")
                log.warning("-" * 100)

            TRAIN_SCORE = row.index(f"TRAIN_{score}") if f"TRAIN_{score}" in row else None
            DEV_SCORE = row.index(f"DEV_{score}") if f"DEV_{score}" in row else None
            TEST_SCORE = row.index(f"TEST_{score}") if f"TEST_{score}" in row else None

            # then get all relevant values from the tsv
            for row in tsvin:
                if TRAIN_SCORE is not None and row[TRAIN_SCORE] != "_":
                    training_curves["train"]["score"].append(float(row[TRAIN_SCORE]))

                if DEV_SCORE is not None and row[DEV_SCORE] != "_":
                    training_curves["dev"]["score"].append(float(row[DEV_SCORE]))

                if TEST_SCORE is not None and row[TEST_SCORE] != "_":
                    training_curves["test"]["score"].append(float(row[TEST_SCORE]))

        return training_curves

    @staticmethod
    def _extract_weight_data(file_name: Union[str, Path]) -> dict:
        if isinstance(file_name, str):
            file_name = Path(file_name)

        weights: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        with open(file_name) as f:
            tsvin = csv.reader(f, delimiter="\t")

            for row in tsvin:
                name = row[WEIGHT_NAME]
                param = row[WEIGHT_NUMBER]
                value = float(row[WEIGHT_VALUE])

                weights[name][param].append(value)

        return weights

    @staticmethod
    def _extract_learning_rate(file_name: Union[str, Path]):
        if isinstance(file_name, str):
            file_name = Path(file_name)

        lrs = []
        losses = []

        with open(file_name) as f:
            tsvin = csv.reader(f, delimiter="\t")
            row = next(tsvin)
            LEARNING_RATE = row.index("LEARNING_RATE")
            TRAIN_LOSS = row.index("TRAIN_LOSS")

            # then get all relevant values from the tsv
            for row in tsvin:
                if row[TRAIN_LOSS] != "_":
                    losses.append(float(row[TRAIN_LOSS]))
                if row[LEARNING_RATE] != "_":
                    lrs.append(float(row[LEARNING_RATE]))

        return lrs, losses

    def plot_weights(self, file_name: Union[str, Path]):
        file_name = Path(file_name)

        weights = self._extract_weight_data(file_name)

        total = len(weights)
        columns = 2
        rows = max(2, int(math.ceil(total / columns)))

        figsize = (4 * columns, 3 * rows)

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
        path = file_name.parent / "weights.png"
        plt.savefig(path, dpi=300)
        log.info(f"Weights plots are saved in {path}")  # to let user know the path of the save plots
        plt.close(fig)

    def plot_training_curves(self, file_name: Union[str, Path], plot_values: list[str] = ["loss", "F1"]):
        file_name = Path(file_name)

        fig = plt.figure(figsize=(15, 10))

        for plot_no, plot_value in enumerate(plot_values):
            training_curves = self._extract_evaluation_data(file_name, plot_value)

            plt.subplot(len(plot_values), 1, plot_no + 1)
            if training_curves["train"]["score"]:
                x = np.arange(0, len(training_curves["train"]["score"]))
                plt.plot(x, training_curves["train"]["score"], label=f"training {plot_value}")
            if training_curves["dev"]["score"]:
                x = np.arange(0, len(training_curves["dev"]["score"]))
                plt.plot(x, training_curves["dev"]["score"], label=f"validation {plot_value}")
            if training_curves["test"]["score"]:
                x = np.arange(0, len(training_curves["test"]["score"]))
                plt.plot(x, training_curves["test"]["score"], label=f"test {plot_value}")
            plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
            plt.ylabel(plot_value)
            plt.xlabel("epochs")

        # save plots
        plt.tight_layout(pad=1.0)
        path = file_name.parent / "training.png"
        plt.savefig(path, dpi=300)
        log.info(f"Loss and F1 plots are saved in {path}")  # to let user know the path of the save plots
        plt.show(block=False)  # to have the plots displayed when user run this module
        plt.close(fig)

    def plot_learning_rate(self, file_name: Union[str, Path], skip_first: int = 10, skip_last: int = 5):
        file_name = Path(file_name)

        lrs, losses = self._extract_learning_rate(file_name)
        lrs = lrs[skip_first:-skip_last] if skip_last > 0 else lrs[skip_first:]
        losses = losses[skip_first:-skip_last] if skip_last > 0 else losses[skip_first:]

        fig, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.0e"))

        # plt.show()

        # save plot
        plt.tight_layout(pad=1.0)
        path = file_name.parent / "learning_rate.png"
        plt.savefig(path, dpi=300)
        log.info(f"Learning_rate plots are saved in {path}")  # to let user know the path of the save plots
        plt.show(block=True)  # to have the plots displayed when user run this module
        plt.close(fig)
