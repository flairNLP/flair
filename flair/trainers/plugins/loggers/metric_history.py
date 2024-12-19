import logging
from collections.abc import Mapping
from typing import Any

from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


default_metrics_to_collect = {
    ("train", "loss"): "train_loss_history",
    ("dev", "score"): "dev_score_history",
    ("dev", "loss"): "dev_loss_history",
}


class MetricHistoryPlugin(TrainerPlugin):
    def __init__(self, metrics_to_collect: Mapping = default_metrics_to_collect) -> None:
        super().__init__()

        self.metric_history: dict[str, list] = {}
        self.metrics_to_collect: Mapping = metrics_to_collect
        for target in self.metrics_to_collect.values():
            self.metric_history[target] = []

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if tuple(record.name) in self.metrics_to_collect:
            target = self.metrics_to_collect[tuple(record.name)]
            self.metric_history[target].append(record.value)

    @TrainerPlugin.hook
    def after_training(self, **kw):
        """Returns metric history."""
        self.trainer.return_values.update(self.metric_history)

    @property
    def attach_to_all_processes(self) -> bool:
        return False

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "metrics_to_collect": dict(self.metrics_to_collect),
        }
