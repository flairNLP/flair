from flair.trainers.plugins.functional import (
    AmpPlugin,
    CheckpointPlugin,
    ModelCardPlugin,
    SchedulerPlugin,
    SWAPlugin,
    WeightExtractorPlugin,
)
from flair.trainers.plugins.loggers import (
    LogFilePlugin,
    LossFilePlugin,
    MetricHistoryPlugin,
    TensorboardLogger,
)

from .base import BasePlugin, Pluggable, TrainerPlugin, TrainingInterrupt
from .metric_records import MetricName, MetricRecord

default_plugins = [
    ModelCardPlugin,
    WeightExtractorPlugin,
    LossFilePlugin,
    MetricHistoryPlugin,
    LogFilePlugin,
]

__all__ = [
    "AmpPlugin",
    "CheckpointPlugin",
    "ModelCardPlugin",
    "SchedulerPlugin",
    "SWAPlugin",
    "WeightExtractorPlugin",
    "LogFilePlugin",
    "LossFilePlugin",
    "MetricHistoryPlugin",
    "TensorboardLogger",
    "BasePlugin",
    "Pluggable",
    "TrainerPlugin",
    "TrainingInterrupt",
    "MetricName",
    "MetricRecord",
]
