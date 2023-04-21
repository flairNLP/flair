import logging

from flair.optim import LinearSchedulerWithWarmup
from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


class LinearSchedulerPlugin(TrainerPlugin):
    """Plugin for LinearSchedulerWithWarmup."""

    def __init__(self, warmup_fraction: float, **kwargs) -> None:
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
        **kw,
    ):
        """Initialize different schedulers, including anneal target for AnnealOnPlateau, batch_growth_annealing, loading schedulers.

        :param dataset_size:
        :param mini_batch_size:
        :param max_epochs:
        :param kw:
        :return:
        """
        # calculate warmup steps
        steps_per_epoch = (dataset_size + mini_batch_size - 1) / mini_batch_size
        num_train_steps = int(steps_per_epoch * max_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_fraction)

        self.scheduler = LinearSchedulerWithWarmup(
            num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer=self.trainer.optimizer
        )

        self.store_learning_rate()

    @TrainerPlugin.hook
    def before_training_epoch(self, **kw):
        """Load state for anneal_with_restarts, batch_growth_annealing, logic for early stopping.

        :param kw:
        :return:
        """
        self.store_learning_rate()
        self.previous_learning_rate = self.current_learning_rate

    @TrainerPlugin.hook
    def after_training_batch(self, **kw):
        """Do the scheduler step if one-cycle or linear decay.

        :param kw:
        :return:
        """
        self.scheduler.step()
        self.store_learning_rate()

    def __str__(self) -> str:
        return f"LinearScheduler | warmup_fraction: '{self.warmup_fraction}'"
