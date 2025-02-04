from collections.abc import Iterable

import torch

from flair.models import MultitaskModel
from flair.nn import DeepNCMDecoder
from flair.trainers.plugins.base import TrainerPlugin


class DeepNCMPlugin(TrainerPlugin):
    """Plugin for training DeepNCMClassifier.

    Handles both multitask and single-task scenarios.
    """

    @property
    def decoders(self) -> Iterable[DeepNCMDecoder]:
        """Iterator over all DeepNCMDecoder decoders in the trainer."""
        model = self.trainer.model

        models = model.tasks.values() if isinstance(model, MultitaskModel) else [model]

        for sub_model in models:
            if hasattr(sub_model, "decoder") and isinstance(sub_model.decoder, DeepNCMDecoder):
                yield sub_model.decoder

    @TrainerPlugin.hook
    def after_training_epoch(self, **kwargs):
        """Reset class counts after each training epoch."""
        for decoder in self.decoders:
            if decoder.mean_update_method == "condensation":
                decoder.class_counts.data = torch.ones_like(decoder.class_counts)

    @TrainerPlugin.hook
    def after_training_batch(self, **kwargs):
        """Update prototypes after each training batch."""
        for decoder in self.decoders:
            decoder.update_prototypes()

    def __str__(self) -> str:
        return "DeepNCMPlugin"
