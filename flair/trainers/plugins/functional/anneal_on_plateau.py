import copy
import inspect
import logging
import os
from typing import List

from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

from flair.optim import LinearSchedulerWithWarmup
from flair.trainers.plugins.base import TrainerPlugin, TrainingInterrupt
from flair.trainers.plugins.metric_records import MetricRecord
from flair.training_utils import AnnealOnPlateau

log = logging.getLogger("flair")


class AnnealingPlugin(TrainerPlugin):
    """
    Plugin for annealing logic in Flair.
    """

    def __init__(self,
                 base_path,
                 min_learning_rate,
                 anneal_factor,
                 patience,
                 initial_extra_patience,
                 anneal_with_restarts,
                 anneal_against_dev_loss):
        super().__init__()

        # path to store the model
        self.base_path = base_path

        # special annealing modes
        self.anneal_with_restarts = anneal_with_restarts

        # determine the min learning rate
        self.min_learning_rate = min_learning_rate

        self.anneal_factor = anneal_factor
        self.patience = patience
        self.initial_extra_patience = initial_extra_patience
        self.anneal_against_dev_loss = anneal_against_dev_loss

    def store_learning_rate(self):
        optimizer = self.trainer.optimizer

        self.current_learning_rate = [group["lr"] for group in optimizer.param_groups]

        self.current_momentum = [
            group["betas"][0] if "betas" in group else group.get("momentum", 0) for group in optimizer.param_groups
        ]

    @TrainerPlugin.hook
    def after_optimizer_setup(
            self,
            train_with_dev,
            optimizer,
            **kw,
    ):
        """
        initialize different schedulers, including anneal target for AnnealOnPlateau, batch_growth_annealing, loading schedulers
        :param train_with_dev:
        :param optimizer:
        :param kw:
        :return:
        """

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev or self.anneal_against_dev_loss else "max"

        # instantiate the scheduler
        self.scheduler = AnnealOnPlateau(factor=self.anneal_factor,
                                         patience=self.patience,
                                         initial_extra_patience=self.initial_extra_patience,
                                         mode=anneal_mode,
                                         verbose=True,
                                         optimizer=self.trainer.optimizer,
                                         )

        self.store_learning_rate()

    @TrainerPlugin.hook
    def before_training_epoch(self, **kw):
        """
        load state for anneal_with_restarts, batch_growth_annealing, logic for early stopping
        :param kw:
        :return:
        """
        self.store_learning_rate()
        self.previous_learning_rate = self.current_learning_rate

        # base_path = self.trainer.base_path

        lr_changed = any(
            [lr != prev_lr for lr, prev_lr in zip(self.current_learning_rate, self.previous_learning_rate)]
        )

        # reload last best model if annealing with restarts is enabled
        if (
                self.anneal_with_restarts
                and lr_changed
                and os.path.exists(self.base_path / "best-model.pt")
        ):
            if self.anneal_with_restarts:
                log.info("resetting to best model")
                self.model.load_state_dict(self.model.load(self.base_path / "best-model.pt").state_dict())

        self.previous_learning_rate = self.current_learning_rate

        # stop training if learning rate becomes too small
        for lr in self.current_learning_rate:
            if lr < self.min_learning_rate:
                raise TrainingInterrupt("learning rate too small - quitting training!")

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch, **kw):
        """
        Logging for bad_epochs
        :param epoch:
        :param kw:
        :return:
        """
        try:
            bad_epochs = self.scheduler.num_bad_epochs

            self.trainer.dispatch(
                "metric_recorded", MetricRecord.scalar(name="bad_epochs", value=bad_epochs, global_step=epoch)
            )
        except AttributeError:
            # dont record anything
            pass

    @TrainerPlugin.hook
    def after_evaluation(self, current_model_is_best, validation_scores, **kw):
        """
        Scheduler step of AnnealOnPlateau
        :param current_model_is_best:
        :param validation_scores:
        :param kw:
        :return:
        """
        if current_model_is_best and isinstance(self.scheduler, AnnealOnPlateau):
            self.scheduler.step(*validation_scores)
