import logging
from typing import Dict, Mapping, Optional

from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


default_metrics_to_collect = {
    ("train", "loss"): "train_loss_history",
    ("dev", "score"): "dev_score_history",
    ("dev", "loss"): "dev_loss_history",
}


class MetricHistoryPlugin(TrainerPlugin):
    def __init__(self, metrics_to_collect: Mapping = default_metrics_to_collect, **kwargs):
        super().__init__(**kwargs)

        self.metric_history: Optional[Dict[str, list]] = None
        self.metrics_to_collect: Mapping = metrics_to_collect

    @TrainerPlugin.hook
    def after_training_setup(self, main_evaluation_metric, **kw):
        """
        initializes history lists for all metrics to collect
        :param main_evaluation_metric:
        :param kw:
        :return:
        """
        self.metric_history = {}

        for target in self.metrics_to_collect.values():
            self.metric_history[target] = list()

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        assert self.metric_history is not None

        try:
            target = self.metrics_to_collect[tuple(record.name)]
            self.metric_history[target].append(record.value)

        except KeyError:
            # metric is not collected
            pass

    @TrainerPlugin.hook
    def after_teardown(self, **kw):
        """
        Returns metric history
        :param kw:
        :return:
        """
        self.trainer.return_values.update(self.metric_history)
