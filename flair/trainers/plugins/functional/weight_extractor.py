from typing import Any

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import WeightExtractor


class WeightExtractorPlugin(TrainerPlugin):
    """Simple Plugin for weight extraction."""

    def __init__(self, base_path) -> None:
        super().__init__()
        self.base_path = base_path
        self.weight_extractor = WeightExtractor(base_path)

    @TrainerPlugin.hook
    def after_training_batch(self, batch_no, epoch, total_number_of_batches, **kw):
        """Extracts weights."""
        modulo = max(1, int(total_number_of_batches / 10))
        iteration = epoch * total_number_of_batches + batch_no

        if (iteration + 1) % modulo == 0:
            self.weight_extractor.extract_weights(self.model.state_dict(), iteration)

    @property
    def attach_to_all_processes(self) -> bool:
        return False

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "base_path": str(self.base_path),
        }
