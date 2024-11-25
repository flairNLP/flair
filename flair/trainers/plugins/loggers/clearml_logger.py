from typing import Any

from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.metric_records import MetricRecord


class ClearmlLoggerPlugin(TrainerPlugin):
    def __init__(self, task_id_or_task: Any):
        if isinstance(task_id_or_task, str):
            self.task_id = task_id_or_task
            self.task = None
        else:
            self.task = task_id_or_task
            self.task_id = self.task.task_id
        super().__init__()

    @property
    def logger(self):
        try:
            import clearml
        except ImportError:
            raise ImportError(
                "Please install clearml 1.11.0 or higher before using the clearml plugin"
                "otherwise you can remove the clearml plugin from the training or model card."
            )
        if self.task is None:
            self.task = clearml.Task.get_task(task_id=self.task_id)
        return self.task.get_logger()

    @TrainerPlugin.hook
    def metric_recorded(self, record: MetricRecord) -> None:
        record_name = ".".join(record.name)

        if record.is_scalar:
            self.logger.report_scalar(record_name, record_name, record.value, record.global_step)
        elif record.is_scalar_list:
            for i, v in enumerate(record.value):
                self.logger.report_scalar(record_name, f"{record_name}_{i}", v, record.global_step)
        elif record.is_string:
            self.logger.report_text(record.value, print_console=False)
        elif record.is_histogram:
            self.logger.report_histogram(record_name, record_name, record.value, record.global_step)

    @property
    def attach_to_all_processes(self) -> bool:
        return False
