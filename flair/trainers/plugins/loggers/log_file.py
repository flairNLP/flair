import logging
from pathlib import Path
from typing import Any

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class LogFilePlugin(TrainerPlugin):
    """Plugin for the training.log file."""

    def __init__(self, base_path) -> None:
        super().__init__()
        self.base_path = base_path
        self.log_handler = add_file_handler(log, Path(base_path) / "training.log")

    @TrainerPlugin.hook("_training_exception", "after_training")
    def close_file_handler(self, **kw):
        self.log_handler.close()
        log.removeHandler(self.log_handler)

    @property
    def attach_to_all_processes(self) -> bool:
        return False

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "base_path": str(self.base_path)}
