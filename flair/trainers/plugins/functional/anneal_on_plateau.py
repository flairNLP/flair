import logging
import os
from typing import Any

from flair.trainers.plugins.base import TrainerPlugin, TrainingInterrupt
from flair.trainers.plugins.metric_records import MetricRecord
from flair.training_utils import AnnealOnPlateau

log = logging.getLogger("flair")


class AnnealingPlugin(TrainerPlugin):
    """Plugin for annealing logic in Flair."""

    def __init__(
        self,
        base_path,
        min_learning_rate,
        anneal_factor,
        patience,
        initial_extra_patience,
        anneal_with_restarts,
    ) -> None:
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
        self.scheduler: AnnealOnPlateau

    def store_learning_rate(self):
        optimizer = self.trainer.optimizer

        self.current_learning_rate = [group["lr"] for group in optimizer.param_groups]

        self.current_momentum = [
            group["betas"][0] if "betas" in group else group.get("momentum", 0) for group in optimizer.param_groups
        ]

    @TrainerPlugin.hook
    def after_setup(
        self,
        train_with_dev,
        optimizer,
        **kw,
    ):
        """Initialize different schedulers, including anneal target for AnnealOnPlateau, batch_growth_annealing, loading schedulers."""
        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"

        # instantiate the scheduler
        self.scheduler: AnnealOnPlateau = AnnealOnPlateau(
            factor=self.anneal_factor,
            patience=self.patience,
            initial_extra_patience=self.initial_extra_patience,
            mode=anneal_mode,
            optimizer=self.trainer.optimizer,
        )

        self.store_learning_rate()

    @TrainerPlugin.hook
    def after_evaluation(self, current_model_is_best, validation_scores, **kw):
        """Scheduler step of AnnealOnPlateau."""
        reduced_learning_rate: bool = self.scheduler.step(*validation_scores)

        self.store_learning_rate()

        bad_epochs = self.scheduler.num_bad_epochs
        if reduced_learning_rate:
            bad_epochs = self.patience + 1
            log.info(
                f" - {bad_epochs} epochs without improvement (above 'patience')"
                f"-> annealing learning_rate to {self.current_learning_rate}"
            )
        else:
            log.info(f" - {bad_epochs} epochs without improvement")

        self.trainer.dispatch(
            "metric_recorded",
            MetricRecord.scalar(name="bad_epochs", value=bad_epochs, global_step=self.scheduler.last_epoch + 1),
        )

        # stop training if learning rate becomes too small
        for lr in self.current_learning_rate:
            if lr < self.min_learning_rate:
                raise TrainingInterrupt("learning rate too small - quitting training!")

        # reload last best model if annealing with restarts is enabled
        if self.anneal_with_restarts and reduced_learning_rate and os.path.exists(self.base_path / "best-model.pt"):
            log.info("resetting to best model")
            self.model.load_state_dict(self.model.load(self.base_path / "best-model.pt").state_dict())

    def __str__(self) -> str:
        return (
            f"AnnealOnPlateau | "
            f"patience: '{self.patience}', "
            f"anneal_factor: '{self.anneal_factor}', "
            f"min_learning_rate: '{self.min_learning_rate}'"
        )

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "base_path": str(self.base_path),
            "min_learning_rate": self.min_learning_rate,
            "anneal_factor": self.anneal_factor,
            "patience": self.patience,
            "initial_extra_patience": self.initial_extra_patience,
            "anneal_with_restarts": self.anneal_with_restarts,
        }
