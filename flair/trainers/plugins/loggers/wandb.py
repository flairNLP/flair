import logging
from typing import Any

from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


class WandbLoggingHandler(logging.Handler):
    def __init__(self, wandb, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.wandb = wandb

    def emit(self, record):
        try:
            # adjust alert level
            if record.level >= logging.ERROR:
                level = self.wandb.AlertLevel.ERROR
            elif record.level >= logging.WARNING:
                level = self.wandb.AlertLevel.WARN
            else:
                level = self.wandb.AlertLevel.INFO

            self.wandb.alert(
                title=f"Alert from {record.module}:{record.lineno}",
                text=self.format(record),
                level=level,
            )

        except Exception:
            self.handleError(record)


class WandbLogger(TrainerPlugin):
    def __init__(self, wandb, emit_alerts=True, alert_level=logging.WARNING) -> None:
        super().__init__()

        self.wandb = wandb
        self.emit_alerts = emit_alerts
        self.alert_level = alert_level
        self._emitted_record_type_warning = False

    @TrainerPlugin.hook
    def after_training_setup(self, **kw):
        if self.emit_alerts:
            self.log_handler = WandbLoggingHandler(self.wandb)
            self.log_handler.setLevel(self.alert_level)

            formatter = logging.Formatter("%(asctime)-15s %(message)s")
            self.log_handler.setFormatter(formatter)
            log.addHandler(self.log_handler)
        else:
            self.log_handler = None

    @TrainerPlugin.hook("_training_exception", "after_teardown")
    def close_file_handler(self, **kw):
        if self.emit_alerts:
            self.log_handler.close()
            log.removeHandler(self.log_handler)

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if record.is_scalar:
            self.wandb.log({record.name: record.value})
        else:
            if not self._emitted_record_type_warning:
                log.warning("Logging anything other than scalars to W&B is currently not supported.")
                self._emitted_record_type_warning = True

    @TrainerPlugin.hook
    def _training_finally(self, **kw):
        self.writer.close()

    @property
    def attach_to_all_processes(self) -> bool:
        return False

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "emit_alerts": self.emit_alerts,
            "alert_level": self.alert_level,
        }
