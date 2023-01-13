import logging
import random

import torch

from flair.data import _len_dataset
from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.metrics.base import MetricBasePlugin, MetricRecord
from flair.training_utils import store_embeddings

log = logging.getLogger("flair")


class BasicPerformancePlugin(MetricBasePlugin):
    def __init__(self):
        super().__init__()
        self.total_train_loss: float = 0.0
        self.total_train_samples: int = 0

        self.train_part_size = None
        self.train_part = None

        self.log_train = None
        self.log_test = None
        self.log_dev = None
        self.log_train_part = None

        self.eval_on_train_shuffle = None
        self.embeddings_storage_mode = None

        self.eval_kw = None

        self.embeddings_storage_mode = None

    @property
    def base_path(self):
        return self.trainer.base_path

    @TrainerPlugin.hook
    def before_training_setup(
        self,
        monitor_train,
        param_selection_mode,
        monitor_test,
        train_with_dev,
        eval_on_train_fraction,
        eval_on_train_shuffle,
        embeddings_storage_mode,
        eval_batch_size,
        mini_batch_size,
        num_workers,
        exclude_labels,
        main_evaluation_metric,
        gold_label_dictionary_for_eval,
        **kw,
    ):
        # determine what splits (train, dev, test) to evaluate and log
        self.log_train = True if monitor_train else False
        self.log_test = True if (not param_selection_mode and self.trainer.corpus.test and monitor_test) else False
        self.log_dev = False if train_with_dev or not self.trainer.corpus.dev else True
        self.log_train_part = (
            True if (eval_on_train_fraction == "dev" or float(eval_on_train_fraction) > 0.0) else False
        )

        self.eval_on_train_shuffle = eval_on_train_shuffle

        self.embeddings_storage_mode = embeddings_storage_mode

        self.eval_kw = {
            "mini_batch_size": eval_batch_size or mini_batch_size,
            "num_workers": num_workers,
            "exclude_labels": exclude_labels,
            "main_evaluation_metric": main_evaluation_metric,
            "gold_label_dictionary": gold_label_dictionary_for_eval,
            "embedding_storage_mode": embeddings_storage_mode,
            "gold_label_type": self.trainer.model.label_type,
        }

    @MetricBasePlugin.hook
    def before_training_epoch(self, epoch, **kw):
        if self.eval_on_train_shuffle:
            train_part_indices = list(range(_len_dataset(self.trainer.corpus.train)))
            random.shuffle(train_part_indices)
            train_part_indices = train_part_indices[: self.train_part_size]
            self.train_part = torch.utils.data.dataset.Subset(self.trainer.corpus.train, train_part_indices)

        self.total_train_loss = 0.0
        self.total_train_samples = 0

        # record current learning rate and momentum
        optimizer = self.trainer.optimizer

        current_learning_rate = [group["lr"] for group in optimizer.param_groups]
        momentum = [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

        if len(current_learning_rate) == 1:
            yield MetricRecord.scalar("learning_rate", current_learning_rate[0], epoch)
            yield MetricRecord.scalar("momentum", momentum[0], epoch)
        else:
            yield MetricRecord.scalar_list("learning_rate", current_learning_rate, epoch)
            yield MetricRecord.scalar_list("momentum", momentum, epoch)

    @TrainerPlugin.hook
    def before_training_batch_backward(self, loss, datapoint_count, **kw):
        self.total_train_samples += datapoint_count
        self.total_train_loss += loss.item()

    @MetricBasePlugin.hook
    def after_training_epoch(self, epoch, **kw):
        if self.total_train_samples != 0:
            train_loss = self.total_train_loss / self.total_train_samples
            yield MetricRecord.scalar(("train", "loss"), train_loss, epoch)

    def flat_dict_items(self, d, composite_key=()):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from self.flat_dict_items(value, composite_key=composite_key + (key,))
            else:
                yield composite_key + (key,), value

    def transform_eval_result(self, result, prefix=(), **kw):
        for key, value in self.flat_dict_items(result.scores, composite_key=prefix):
            try:
                yield MetricRecord.scalar(name=key, value=float(value), **kw)
            except TypeError:
                if isinstance(value, list):
                    yield MetricRecord.scalar_list(name=key, value=value, **kw)
                elif isinstance(value, torch.Tensor):
                    yield MetricRecord.histogram(name=key, value=value, **kw)
                else:
                    value = str(value)
                    yield MetricRecord.string(name=key, value=value, **kw)

        yield MetricRecord.string(identifier=prefix + ("detailed_result",), value=result.detailed_results, **kw)

    # yielded metric values are automatically published via the trainer
    @MetricBasePlugin.hook
    def evaluation(self, epoch, **kw):
        # evaluate on train / dev / test split depending on training settings
        if self.log_train:
            train_eval_result = self.model.evaluate(self.corpus.train, **self.eval_kw)

            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.train, self.embeddings_storage_mode)

            yield from self.transform_eval_result(train_eval_result, "train_eval")

        if self.log_train_part:
            train_part_eval_result = self.model.evaluate(self.train_part, **self.eval_kw)

            log.info(
                f"TRAIN_SPLIT : loss {train_part_eval_result.loss}"
                f' - {self.eval_kw["main_evaluation_metric"][1]}'
                f' ({self.eval_kw["main_evaluation_metric"][0]})'
                f" {round(train_part_eval_result.main_score, 4)}"
            )

            yield from self.transform_eval_result(train_part_eval_result, "train_part")

        if self.log_dev:
            assert self.corpus.dev
            dev_eval_result = self.model.evaluate(self.corpus.dev, out_path=self.base_path / "dev.tsv", **self.eval_kw)

            log.info(
                f"DEV : loss {dev_eval_result.loss}"
                f' - {self.eval_kw["main_evaluation_metric"][1]}'
                f' ({self.eval_kw["main_evaluation_metric"][0]})'
                f"  {round(dev_eval_result.main_score, 4)}"
            )

            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.dev, self.embeddings_storage_mode)

            yield from self.transform_eval_result(train_part_eval_result, "dev")

        if self.log_test:
            assert self.corpus.test
            test_eval_result = self.model.evaluate(
                self.corpus.test,
                gold_label_type=self.model.label_type,
                out_path=self.base_path / "test.tsv",
                **self.eval_kw,
            )

            log.info(
                f"TEST : loss {test_eval_result.loss} -"
                f' {self.eval_kw["main_evaluation_metric"][1]}'
                f' ({self.eval_kw["main_evaluation_metric"][0]}) '
                f" {round(test_eval_result.main_score, 4)}"
            )

            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.test, self.embeddings_storage_mode)

            yield from self.transform_eval_result(test_eval_result, "test")

        @TrainerPlugin.hook
        def collecting_train_return_values(self, **kw):
            # test best model if test data is present
            if self.corpus.test and not self.train_with_test:

                final_score = self.final_test(
                    base_path=self.base_path,
                    eval_mini_batch_size=self.eval_kw["mini_batch_size"],
                    num_workers=self.eval_kw["num_workers"],
                    main_evaluation_metric=self.eval_kw["main_evaluation_metric"],
                    gold_label_dictionary_for_eval=self.eval_kw["gold_label_dictionary_for_eval"],
                    exclude_labels=self.eval_kw["exclude_labels"],
                )
            else:
                final_score = 0
                log.info("Test data not provided setting final score to 0")

            # dicts returned by after_training callbacks get collected, joined
            # together and are returned by the train function
            return {"final_score": final_score}
