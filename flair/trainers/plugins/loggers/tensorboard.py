import logging
import os

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import log_line

log = logging.getLogger("flair")


class TensorboardLogger(TrainerPlugin):
    def __init__(self, log_dir=None, comment="", tracked_metrics=()):
        """
        :param log_dir: Directory into which tensorboard log files will be written  # noqa: E501
        :param tracked_metrics: List of tuples that specify which metrics (in addition to the main_score) shall be plotted in tensorboard, could be [("macro avg", 'f1-score'), ("macro avg", 'precision')] for example  # noqa: E501
        """
        super().__init__()
        self.log_dir = log_dir
        self.comment = comment
        self.tracked_metrics = tracked_metrics
        self.writer = None

        self._warned = False

    @TrainerPlugin.hook
    def after_data_setup(self, use_tensorboard, **kw):
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                if self.log_dir is not None and not os.path.exists(self.log_dir):
                    os.mkdir(self.log_dir)

                self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)

                log.info(f"tensorboard logging path is {self.log_dir}")

            except ImportError:
                log_line(log)
                log.warning("ATTENTION! PyTorch >= 1.1.0 and pillow are required" "for TensorBoard support!")
                log_line(log)

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if self.writer is not None:
            # TODO: check if metric is in tracked metrics
            if record.is_scalar:
                self.writer.add_scalar(str(record.name), record.value, record.global_step, walltime=record.walltime)
            else:
                if not self._warned:
                    log.warning("Logging anything other than scalars to TensorBoard is currently not supported.")
                    self._warned = True

    @TrainerPlugin.hook
    def _training_finally(self, **kw):
        if self.writer is not None:
            self.writer.close()
