from datetime import datetime
from typing import Any, Optional, Union

from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.metric_records import MetricName
from flair.training_utils import init_output_file


class LossFilePlugin(TrainerPlugin):
    """Plugin that manages the loss.tsv file output."""

    def __init__(
        self, base_path, epoch: int, metrics_to_collect: Optional[dict[Union[tuple, str], str]] = None
    ) -> None:
        super().__init__()

        self.first_epoch = epoch + 1
        # prepare loss logging file and set up header
        self.loss_txt = init_output_file(base_path, "loss.tsv")
        self.base_path = base_path

        # set up all metrics to collect
        self.metrics_to_collect = metrics_to_collect
        if self.metrics_to_collect is not None:
            metrics_to_collect = self.metrics_to_collect
        else:
            metrics_to_collect = {
                "loss": "LOSS",
                ("micro avg", "precision"): "PRECISION",
                ("micro avg", "recall"): "RECALL",
                ("micro avg", "f1-score"): "F1",
                "accuracy": "ACCURACY",
            }

        # set up headers
        self.headers = {
            # name: HEADER
            MetricName("epoch"): "EPOCH",
            MetricName("timestamp"): "TIMESTAMP",
            MetricName("bad_epochs"): "BAD_EPOCHS",
            MetricName("learning_rate"): "LEARNING_RATE",
        }

        # Add all potentially relevant metrics. If a metric is not published
        # after the first epoch (when the header is written), the column is
        # removed at that point.
        for prefix in ["train", "train_sample", "dev", "test"]:
            for name, header in metrics_to_collect.items():
                metric_name = MetricName(name)

                if prefix == "train" and metric_name != "loss":
                    metric_name = "train_eval" + metric_name
                else:
                    metric_name = prefix + metric_name

                self.headers[metric_name] = f"{prefix.upper()}_{header}"

        # initialize the first log line
        self.current_row: Optional[dict[MetricName, str]] = None

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "base_path": str(self.base_path),
            "metrics_to_collect": self.metrics_to_collect,
        }

    @TrainerPlugin.hook
    def before_training_epoch(self, epoch, **kw):
        """Get the current epoch for loss file logging."""
        self.current_row = {MetricName("epoch"): epoch}

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        """Add the metric of a record to the current row."""
        if record.name in self.headers and self.current_row is not None:
            if record.name == "learning_rate" and not record.is_scalar:
                # record is a list of scalars
                value = ",".join([f"{lr:.4f}" for lr in record.value])
            elif record.is_scalar and isinstance(record.value, int):
                value = str(record.value)
            else:
                assert record.is_scalar

                value = f"{record.value:.4f}"

            self.current_row[record.name] = value

    @TrainerPlugin.hook
    def after_evaluation(self, epoch, **kw):
        """This prints all relevant metrics."""
        if self.loss_txt is not None:
            self.current_row[MetricName("timestamp")] = f"{datetime.now():%H:%M:%S}"

            # output log file
            with open(self.loss_txt, "a") as f:
                # remove columns where no value was found on the first epoch (could be != 1 if training was resumed)
                if epoch == self.first_epoch:
                    for k in list(self.headers.keys()):
                        if k not in self.current_row:
                            del self.headers[k]

                # make headers on epoch 1
                if epoch == 1:
                    # write header
                    f.write("\t".join(self.headers.values()) + "\n")

                for col in self.headers:
                    assert col in self.current_row, str(col) + "   " + str(self.current_row.keys())

                assert all(col in self.current_row for col in self.headers)

                f.write("\t".join([str(self.current_row[col]) for col in self.headers]) + "\n")

            self.current_row = {}

    @property
    def attach_to_all_processes(self) -> bool:
        return False
