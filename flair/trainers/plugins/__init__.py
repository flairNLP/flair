from flair.trainers.plugins.functional import (
    AmpPlugin,
    CheckpointPlugin,
    ModelCardPlugin,
    RegularLoggingPlugin,
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
from flair.trainers.plugins.metrics import BasicEvaluationPlugin, TrainingBehaviorPlugin

from .base import BasePlugin, Pluggable, TrainerPlugin, TrainingInterrupt

default_plugins = [
    BasicEvaluationPlugin,
    TrainingBehaviorPlugin,
    RegularLoggingPlugin,
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
    "RegularLoggingPlugin",
    "SchedulerPlugin",
    "SWAPlugin",
    "WeightExtractorPlugin",
    "LogFilePlugin",
    "LossFilePlugin",
    "MetricHistoryPlugin",
    "TensorboardLogger",
    "BasicEvaluationPlugin",
    "TrainingBehaviorPlugin",
    "BasePlugin",
    "Pluggable",
    "TrainerPlugin",
    "TrainingInterrupt",
]
