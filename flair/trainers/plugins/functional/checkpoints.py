import logging
from typing import Any

from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


class CheckpointPlugin(TrainerPlugin):
    def __init__(
        self,
        save_model_each_k_epochs,
        save_optimizer_state,
        base_path,
    ) -> None:
        super().__init__()
        self.save_optimizer_state = save_optimizer_state
        self.save_model_each_k_epochs = save_model_each_k_epochs
        self.base_path = base_path

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch, **kw):
        """Saves the model each k epochs."""
        if self.save_model_each_k_epochs > 0 and epoch % self.save_model_each_k_epochs == 0:
            log.info(
                f"Saving model at current epoch since 'save_model_each_k_epochs={self.save_model_each_k_epochs}' "
                f"was set"
            )
            model_name = "model_epoch_" + str(epoch) + ".pt"
            self.model.save(self.base_path / model_name, checkpoint=self.save_optimizer_state)

    @property
    def attach_to_all_processes(self) -> bool:
        return False

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "base_path": str(self.base_path),
            "save_model_each_k_epochs": self.save_model_each_k_epochs,
            "save_optimizer_state": self.save_optimizer_state,
        }
