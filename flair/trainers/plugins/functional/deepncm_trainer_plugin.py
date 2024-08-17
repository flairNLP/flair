import torch

from flair.models import DeepNCMClassifier, MultitaskModel
from flair.trainers.plugins.base import TrainerPlugin


class DeepNCMPlugin(TrainerPlugin):
    """Plugin for training DeepNCMClassifier.

    Handles both multitask and single-task scenarios.
    """

    def _process_models(self, operation: str):
        """Process updates for all DeepNCMClassifier models in the trainer.

        Args:
            operation (str): The operation to perform ('condensation' or 'update')
        """
        model = self.trainer.model

        models = model.tasks.values() if isinstance(model, MultitaskModel) else [model]

        for sub_model in models:
            if isinstance(sub_model, DeepNCMClassifier):
                if operation == "condensation" and sub_model.mean_update_method == "condensation":
                    sub_model.class_counts.data = torch.ones_like(sub_model.class_counts)
                elif operation == "update":
                    sub_model.update_prototypes()

    @TrainerPlugin.hook
    def after_training_epoch(self, **kwargs):
        """Update prototypes after each training epoch."""
        self._process_models("condensation")

    @TrainerPlugin.hook
    def after_training_batch(self, **kwargs):
        """Update prototypes after each training batch."""
        self._process_models("update")

    def __str__(self) -> str:
        return "DeepNCMPlugin"
