import logging
import os

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import log_line

log = logging.getLogger("flair")


class TensorboardLogger(TrainerPlugin):
    """Plugin that takes care of tensorboard logging."""

    def __init__(self, log_dir=None, comment="", tracked_metrics=()) -> None:
        """Initializes the TensorboardLogger.

        :param log_dir: Directory into which tensorboard log files will be written  # noqa: E501
        :param tracked_metrics: List of tuples that specify which metrics (in addition to the main_score) shall be plotted in tensorboard, could be [("macro avg", 'f1-score'), ("macro avg", 'precision')] for example  # noqa: E501
        """
        super().__init__()
        self.comment = comment
        self.tracked_metrics = tracked_metrics

        try:
            from torch.utils.tensorboard import SummaryWriter

            if log_dir is not None and not os.path.exists(log_dir):
                os.mkdir(log_dir)

            self.writer = SummaryWriter(log_dir=log_dir, comment=self.comment)

            log.info(f"tensorboard logging path is {log_dir}")

        except ImportError:
            log_line(log)
            log.warning("ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!")
            log_line(log)

        self._warned = False

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        assert self.writer is not None
        # TODO: check if metric is in tracked metrics
        if record.is_scalar:
            self.writer.add_scalar(str(record.name), record.value, record.global_step, walltime=record.walltime)
        else:
            if not self._warned:
                log.warning("Logging anything other than scalars to TensorBoard is currently not supported.")
                self._warned = True

    @TrainerPlugin.hook
    def _training_finally(self, **kw):
        """Closes the writer.

        :param kw:
        :return:
        """
        assert self.writer is not None
        self.writer.close()
