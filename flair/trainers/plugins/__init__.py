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
from flair.trainers.plugins.metrics import BasicPerformancePlugin

from .base import BasePlugin, Pluggable, TrainerPlugin, TrainingInterrupt

default_plugins = [
    BasicPerformancePlugin,
    AmpPlugin,
    CheckpointPlugin,
    ModelCardPlugin,
    SchedulerPlugin,
    SWAPlugin,
    WeightExtractorPlugin,
    LossFilePlugin,
    MetricHistoryPlugin,
    TensorboardLogger,
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
    "BasicPerformancePlugin",
    "BasePlugin",
    "Pluggable",
    "TrainerPlugin",
    "TrainingInterrupt",
]
