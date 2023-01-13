import logging

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class LogFilePlugin(TrainerPlugin):
    @TrainerPlugin.hook
    def before_training_setup(self, create_file_logs, base_path, **kw):
        self.create_file_logs = create_file_logs
        self.base_path = base_path

    @TrainerPlugin.hook
    def before_training_loop(self, **kw):
        if self.create_file_logs:
            self.log_handler = add_file_handler(log, self.base_path / "training.log")
        else:
            self.log_handler = None

    @TrainerPlugin.hook("_training_finally")
    def close_file_handler(self, **kw):
        if self.create_file_logs:
            self.log_handler.close()
            log.removeHandler(self.log_handler)
