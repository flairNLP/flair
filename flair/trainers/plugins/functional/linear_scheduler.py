import logging
from typing import Any

import torch.distributed

from flair.optim import LinearSchedulerWithWarmup
from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


class LinearSchedulerPlugin(TrainerPlugin):
    """Plugin for LinearSchedulerWithWarmup."""

    def __init__(self, warmup_fraction: float) -> None:
        super().__init__()

        self.warmup_fraction = warmup_fraction

    def store_learning_rate(self):
        optimizer = self.trainer.optimizer

        self.current_learning_rate = [group["lr"] for group in optimizer.param_groups]

        self.current_momentum = [
            group["betas"][0] if "betas" in group else group.get("momentum", 0) for group in optimizer.param_groups
        ]

    @TrainerPlugin.hook
    def after_setup(
        self,
        dataset_size,
        mini_batch_size,
        max_epochs,
        **kwargs,
    ):
        """Initialize different schedulers, including anneal target for AnnealOnPlateau, batch_growth_annealing, loading schedulers."""
        # calculate warmup steps
        num_processes = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        steps_per_epoch = (dataset_size + mini_batch_size - 1) / mini_batch_size / num_processes
        num_train_steps = int(steps_per_epoch * max_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_fraction)

        self.scheduler = LinearSchedulerWithWarmup(
            num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer=self.trainer.optimizer
        )

        self.store_learning_rate()

    @TrainerPlugin.hook
    def before_training_epoch(self, **kwargs):
        """Load state for anneal_with_restarts, batch_growth_annealing, logic for early stopping."""
        self.store_learning_rate()
        self.previous_learning_rate = self.current_learning_rate

    @TrainerPlugin.hook
    def after_training_batch(self, optimizer_was_run: bool, **kwargs):
        """Do the scheduler step if one-cycle or linear decay."""
        # skip if no optimization has happened.
        if not optimizer_was_run:
            return
        self.scheduler.step()
        self.store_learning_rate()

    def __str__(self) -> str:
        return f"LinearScheduler | warmup_fraction: '{self.warmup_fraction}'"

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "warmup_fraction": self.warmup_fraction,
        }
