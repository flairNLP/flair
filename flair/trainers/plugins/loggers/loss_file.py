from datetime import datetime
from typing import Dict, Tuple

from flair.nn import Classifier
from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.metrics import MetricName
from flair.training_utils import init_output_file


class LossFilePlugin(TrainerPlugin):
    def __init__(self, metrics_to_collect: Dict[Tuple, str] = None, **kwargs):
        super().__init__(**kwargs)

        self.loss_txt = None

        self.current_row = None
        self.headers = None
        self.metrics_to_collect = metrics_to_collect

    @TrainerPlugin.hook
    def before_training_setup(self, create_loss_file, base_path, **kw):
        # prepare loss logging file and set up header
        if create_loss_file:
            self.loss_txt = init_output_file(base_path, "loss.tsv")
        else:
            self.loss_txt = None

        self.headers = {
            # name: HEADER
            "epoch": "EPOCH",
            "timestamp": "TIMESTAMP",
            "bad_epochs": "BAD_EPOCHS",
            "learning_rate": "LEARNING_RATE",
        }

        if self.metrics_to_collect is not None:
            metrics_to_collect = self.metrics_to_collect
        elif isinstance(self.trainer.model, Classifier):
            metrics_to_collect = {
                "loss": "LOSS",
                ("micro avg", "precision"): "PRECISION",
                ("micro avg", "recall"): "RECALL",
                ("micro avg", "f1-score"): "ACCURACY",
            }
        else:
            metrics_to_collect = {
                "loss": "LOSS",
            }

        # Add all potentially relevant metrics. If a metric is not published
        # after the first epoch (when the header is written), the column is
        # removed at that point.
        for prefix in ["train", "train_part", "dev", "test"]:
            for name, header in metrics_to_collect.items():
                name = MetricName(name)

                if prefix == "train" and name != "loss":
                    name = "train_eval" + name
                else:
                    name = prefix + name

                self.headers[name] = f"{prefix.upper()}_{header}"

    @TrainerPlugin.hook
    def before_training_epoch(self, epoch, **kw):
        self.recorded_values = {"epoch": epoch}

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if record.name in self.headers and self.current_row is not None:
            if record.name == "learning_rate" and not record.is_scalar:
                # record is a list of scalars
                value = ",".join([f"{lr:.4f}" for lr in record.value])
            else:
                assert record.is_scalar

                value = f"{record.value:.4f}"

            self.current_row[record.name] = value

    @TrainerPlugin.hook
    def after_evaluation(self, epoch, **kw):
        if self.loss_txt is not None:
            self.current_row["timestamp"] = f"{datetime.datetime.now():%H:%M:%S}"

            # output log file
            with open(self.loss_txt, "a") as f:
                # make headers on first epoch
                if epoch == 1:
                    # delete all headers were no value was recorded
                    for k in self.headers.keys():
                        if k not in self.current_row:
                            del self.headers[k]

                    # write header
                    f.write("\t".join(self.headers.values()))

                for col in self.headers.keys():
                    f.write("\t".join([self.current_row[col]]))

            self.current_row = {}
