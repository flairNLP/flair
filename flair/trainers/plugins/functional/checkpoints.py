import logging
import torch
import json
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
        """Saves the model each k epochs.

        :param epoch:
        :param kw:
        :return:
        """
        if self.save_model_each_k_epochs > 0 and epoch % self.save_model_each_k_epochs == 0:
            log.info(
                f"Saving model at current epoch since 'save_model_each_k_epochs={self.save_model_each_k_epochs}' "
                f"was set"
            )
            model_name = "model_epoch_" + str(epoch) + ".pt"
            self.model.save(self.base_path / model_name, checkpoint=self.save_optimizer_state)

            if self.save_optimizer_state:
                #checkpoint_name = "model_epoch_" + str(epoch) + "_optimizer.pt"
                #optimizer_state_dict = self.trainer.optimizer.state_dict()
                #torch.save(optimizer_state_dict, self.base_path / checkpoint_name)

                scheduler_name = "model_epoch_" + str(epoch) + "_scheduler.pt"
                torch.save(self.trainer.plugins[0].scheduler, self.base_path / scheduler_name)


                # scheduler_name = "model_epoch_" + str(epoch) + "_scheduler.json"
                # scheduler_values = {"step_count": self.trainer.plugins[0].scheduler._step_count,
                #                     "last_lr": self.trainer.plugins[0].scheduler._last_lr[-1]
                #                    }
                # with open(self.base_path / scheduler_name, "w") as outfile:
                #     json.dump(scheduler_values, outfile)
