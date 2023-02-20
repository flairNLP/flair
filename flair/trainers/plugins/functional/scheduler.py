import copy
import inspect
import logging
import os
from typing import List

from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

from flair.optim import LinearSchedulerWithWarmup
from flair.trainers.plugins.base import TrainerPlugin, TrainingInterrupt
from flair.trainers.plugins.functional.best_model import BestModelPlugin
from flair.training_utils import AnnealOnPlateau

from flair.trainers.plugins.metrics.base import MetricRecord


log = logging.getLogger("flair")


class SchedulerPlugin(TrainerPlugin):
    dependencies = (BestModelPlugin,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.initial_learning_rate: List = None
        self.current_learning_rate: List = None
        self.current_momentum: List = None

        self.relevant_metric = None

        self.scheduler_kw = None

        self.dataset_size = None

        self.scheduler = None

        self.anneal_with_prestarts = None
        self.anneal_with_restarts = None

        self.last_epoch_model_state_dict = None
        self.batch_growth_annealing = None

    def store_learning_rate(self):
        optimizer = self.trainer.optimizer

        self.current_learning_rate = [group["lr"] for group in optimizer.param_groups]

        self.current_momentum = [
            group["betas"][0] if "betas" in group else group.get("momentum", 0) for group in optimizer.param_groups
        ]

    @TrainerPlugin.hook
    def before_training_setup(self, scheduler, batch_growth_annealing, **kw):
        if isinstance(scheduler, OneCycleLR) and batch_growth_annealing:
            raise ValueError("Batch growth with OneCycle policy is not implemented.")

    @TrainerPlugin.hook
    def after_optimizer_setup(
        self,
        dataset_size,
        min_learning_rate,
        train_with_dev,
        anneal_against_dev_loss,
        scheduler,
        cycle_momentum,
        warmup_fraction,
        anneal_factor,
        patience,
        initial_extra_patience,
        scheduler_state_dict,
        batch_growth_annealing,
        mini_batch_size,
        max_epochs,
        epoch,
        anneal_with_prestarts,
        anneal_with_restarts,
        **kw,
    ):
        optimizer = self.trainer.optimizer

        self.initial_learning_rate = [group["lr"] for group in optimizer.param_groups]

        if not isinstance(min_learning_rate, list):
            min_learning_rate = [min_learning_rate] * len(self.initial_learning_rate)

        for i, lr in enumerate(self.initial_learning_rate):
            if lr < min_learning_rate[i]:
                min_learning_rate[i] = lr / 10

        self.min_learning_rate = min_learning_rate
        self.batch_growth_annealing = batch_growth_annealing

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev or anneal_against_dev_loss else "max"

        self.scheduler = scheduler

        if inspect.isclass(scheduler):
            if scheduler == OneCycleLR:
                self.scheduler_kw = dict(
                    max_lr=self.current_learning_rate,
                    steps_per_epoch=dataset_size // mini_batch_size + 1,
                    epochs=max_epochs - epoch,
                    # if we load a checkpoint, we have already trained for epoch
                    pct_start=0.0,
                    cycle_momentum=cycle_momentum,
                )
            elif scheduler == LinearSchedulerWithWarmup:
                steps_per_epoch = (dataset_size + mini_batch_size - 1) / mini_batch_size
                num_train_steps = int(steps_per_epoch * max_epochs)
                num_warmup_steps = int(num_train_steps * warmup_fraction)

                self.scheduler_kw = dict(
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                )
            else:
                self.scheduler_kw = dict(
                    factor=anneal_factor,
                    patience=patience,
                    initial_extra_patience=initial_extra_patience,
                    mode=anneal_mode,
                    verbose=True,
                )

        self.scheduler_state_dict = scheduler_state_dict

        self.store_learning_rate()

        # if scheduler is passed as a class, instantiate
        if inspect.isclass(self.scheduler):
            self.scheduler = self.scheduler(optimizer, **self.scheduler_kw)

        # load existing scheduler state dictionary if it exists
        if self.scheduler_state_dict:
            self.scheduler.load_state_dict(self.scheduler_state_dict)

        self.log_bad_epochs = isinstance(scheduler, AnnealOnPlateau)

        self.anneal_with_prestarts = anneal_with_prestarts
        self.anneal_with_restarts = anneal_with_restarts

    @TrainerPlugin.hook
    def before_training_loop(self, **kw):
        self.store_learning_rate()
        self.previous_learning_rate = self.current_learning_rate

    @TrainerPlugin.hook
    def before_training_epoch(self, **kw):
        self.store_learning_rate()

        base_path = self.trainer.base_path

        if self.anneal_with_prestarts:
            self.last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

        lr_changed = any(
            [lr != prev_lr for lr, prev_lr in zip(self.current_learning_rate, self.previous_learning_rate)]
        )

        if lr_changed and self.batch_growth_annealing:
            self.trainer.mini_batch_size *= 2

        # reload last best model if annealing with restarts is enabled
        if (
            (self.anneal_with_restarts or self.anneal_with_prestarts)
            and lr_changed
            and os.path.exists(base_path / "best-model.pt")
        ):
            if self.anneal_with_restarts:
                log.info("resetting to best model")
                self.model.load_state_dict(self.model.load(base_path / "best-model.pt").state_dict())
            if self.anneal_with_prestarts:
                log.info("resetting to pre-best model")
                self.model.load_state_dict(self.model.load(base_path / "pre-best-model.pt").state_dict())

        self.previous_learning_rate = self.current_learning_rate

        all_lrs_too_small = all([lr < min_lr for lr, min_lr in zip(self.current_learning_rate, self.min_learning_rate)])

        # stop training if learning rate becomes too small
        if not isinstance(self.scheduler, (OneCycleLR, LinearSchedulerWithWarmup)) and all_lrs_too_small:
            raise TrainingInterrupt("learning rate too small - quitting training!")

    @TrainerPlugin.hook
    def after_training_batch(self, **kw):
        # do the scheduler step if one-cycle or linear decay
        if isinstance(self.scheduler, (OneCycleLR, LinearSchedulerWithWarmup)):
            self.scheduler.step()
            self.store_learning_rate()

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch, **kw):
        if self.log_bad_epochs:
            try:
                bad_epochs = self.scheduler.num_bad_epochs

                self.trainer.dispatch("metric_recorded", MetricRecord.scalar(
                    name="bad_epochs", value=bad_epochs, global_step=epoch))
            except AttributeError:
                # dont record anything
                pass

    @TrainerPlugin.hook
    def best_model(self, primary_value, auxiliary_value, **kw):
        if isinstance(self.scheduler, AnnealOnPlateau):
            self.scheduler.step(primary_value, auxiliary_value)
