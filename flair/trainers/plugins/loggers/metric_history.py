import logging
from typing import Dict, Mapping, Optional

from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")

default_metrics_to_collect = {
    "train/loss": "dev_score_history",
    "dev/score": "train_loss_history",
    "dev/loss": "dev_loss_history",
}


class MetricHistoryPlugin(TrainerPlugin):
    def __init__(self, metrics_to_collect: Mapping = default_metrics_to_collect, **kwargs):
        super().__init__(**kwargs)

        self.metric_history: Optional[Dict[str, list]] = None
        self.metrics_to_collect = metrics_to_collect

    @TrainerPlugin.hook
    def after_training_setup(self, **kw):
        self.metric_history = {}

        for target in self.metrics_to_collect.values():
            self.metric_history[target] = list()

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        assert self.metric_history is not None

        try:
            target = self.metrics_to_collect[record.name]
            self.metric_history[target].append(record.value)

        except KeyError:
            # metric is not collected
            pass

    @TrainerPlugin.hook
    def collecting_train_return_values(self, **kw):
        return self.metric_history
