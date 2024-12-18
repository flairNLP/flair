import torch

from flair.models import MultitaskModel
from flair.nn import DeepNCMDecoder
from flair.trainers.plugins.base import TrainerPlugin


class DeepNCMPlugin(TrainerPlugin):
    """Plugin for training DeepNCMClassifier.

    Handles both multitask and single-task scenarios.
    """

    def _process_models(self, operation: str):
        """Process updates for all DeepNCMDecoder decoders in the trainer.

        Args:
            operation (str): The operation to perform ('condensation' or 'update')
        """
        model = self.trainer.model

        models = model.tasks.values() if isinstance(model, MultitaskModel) else [model]

        for sub_model in models:
            if hasattr(sub_model, "decoder") and isinstance(sub_model.decoder, DeepNCMDecoder):
                if operation == "condensation" and sub_model.decoder.mean_update_method == "condensation":
                    sub_model.decoder.class_counts.data = torch.ones_like(sub_model.decoder.class_counts)
                elif operation == "update":
                    sub_model.decoder.update_prototypes()

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
