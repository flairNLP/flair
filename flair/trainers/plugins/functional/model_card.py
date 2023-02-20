from pathlib import Path
from typing import Any, Dict

import torch

import flair
from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.functional.scheduler import SchedulerPlugin


class ModelCardPlugin(TrainerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_card: Dict[str, Any] = None

    @TrainerPlugin.hook
    def before_training_setup(self, **training_parameters):
        # create a model card for this model with Flair and PyTorch version
        self.model_card = {
            "flair_version": flair.__version__,
            "pytorch_version": torch.__version__,
        }

        # record Transformers version if library is loaded
        try:
            import transformers

            self.model_card["transformers_version"] = transformers.__version__
        except ImportError:
            pass

        # remember all parameters used in train() call
        self.model_card["training_parameters"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in training_parameters.items()
        }

        # remember all activated plugins
        self.model_card["plugins"] = [plugin.__class__ for plugin in self.trainer.plugins]

        # add model card to model
        self.trainer.model.model_card = self.model_card

    @TrainerPlugin.hook
    def before_training_epoch(self, epoch, **kw):
        # update epoch in model card
        self.model_card["training_parameters"]["epoch"] = epoch

    @TrainerPlugin.hook
    def after_training_setup(self, **kw):
        # update optimizer and scheduler in model card
        self.model_card["training_parameters"]["optimizer"] = self.trainer.optimizer

        for plugin in self.trainer.plugins:
            if isinstance(plugin, SchedulerPlugin):
                self.model_card["training_parameters"]["scheduler"] = plugin.scheduler
