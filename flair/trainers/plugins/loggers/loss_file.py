from datetime import datetime
from typing import Dict, Optional, Tuple

from flair.nn import Classifier
from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.metrics import MetricName
from flair.training_utils import init_output_file


class LossFilePlugin(TrainerPlugin):
    """
    Plugin that manages the loss.tsv file output
    """

    def __init__(self, metrics_to_collect: Dict[Tuple, str] = None, **kwargs):
        super().__init__(**kwargs)

        self.loss_txt = None

        self.current_row: Optional[Dict[MetricName, str]] = None
        self.headers = None
        self.metrics_to_collect = metrics_to_collect

    @TrainerPlugin.hook
    def before_training_setup(self, create_loss_file, base_path, epoch, **kw):
        """
        Prepare loss file, and header for all metrics to collect
        :param create_loss_file:
        :param base_path:
        :param epoch:
        :param kw:
        :return:
        """

        # prepare loss logging file and set up header
        if create_loss_file:
            self.loss_txt = init_output_file(base_path, "loss.tsv")
        else:
            self.loss_txt = None

        self.first_epoch = epoch + 1

        self.headers = {
            # name: HEADER
            MetricName("epoch"): "EPOCH",
            MetricName("timestamp"): "TIMESTAMP",
            MetricName("bad_epochs"): "BAD_EPOCHS",
            MetricName("learning_rate"): "LEARNING_RATE",
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
        """
        Get the current epoch for loss file logging
        :param epoch:
        :param kw:
        :return:
        """
        self.current_row = {MetricName("epoch"): epoch}

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        """
        TODO: I don't really understand this
        :param record:
        :return:
        """
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
        """
        This somehow prints all relevant metrics (TODO: I don't really understand how)
        :param epoch:
        :param kw:
        :return:
        """
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
                    f.write("\t".join(self.headers.values()))

                for col in self.headers.keys():
                    assert col in self.current_row, str(col) + "   " + str(self.current_row.keys())

                assert all(col in self.current_row for col in self.headers.keys())

                f.write("\t".join([str(self.current_row[col]) for col in self.headers.keys()]))

            self.current_row = {}
