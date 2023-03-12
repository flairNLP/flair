import logging

from flair.trainers.plugins.base import TrainerPlugin
from flair.trainers.plugins.functional.best_model import BestModelPlugin

log = logging.getLogger("flair")


class CheckpointPlugin(TrainerPlugin):
    dependencies = (BestModelPlugin,)

    @TrainerPlugin.hook
    def before_training_setup(
        self,
        save_model_each_k_epochs,
        checkpoint,
        param_selection_mode,
        save_final_model,
        save_optimizer_state,
        train_with_dev,
        anneal_with_restarts,
        anneal_with_prestarts,
        use_final_model_for_eval,
        **kw,
    ):
        """
        Determines how and what is saved
        :param save_model_each_k_epochs:
        :param checkpoint:
        :param param_selection_mode:
        :param save_final_model:
        :param save_optimizer_state:
        :param train_with_dev:
        :param anneal_with_restarts:
        :param anneal_with_prestarts:
        :param use_final_model_for_eval:
        :param kw:
        :return:
        """
        self.save_model_each_k_epochs = save_model_each_k_epochs
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint = checkpoint
        self.param_selection_mode = param_selection_mode
        self.save_final_model = save_final_model

        self.save_best_model = (
            (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
            and not param_selection_mode
            and not use_final_model_for_eval
        )

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch, **kw):
        """
        Executes save_model_each_k_epochs
        :param epoch:
        :param kw:
        :return:
        """
        if self.save_model_each_k_epochs > 0 and epoch % self.save_model_each_k_epochs == 0:
            log.info("saving model of current epoch")
            model_name = "model_epoch_" + str(epoch) + ".pt"
            self.model.save(self.trainer.base_path / model_name, checkpoint=self.save_optimizer_state)

    @TrainerPlugin.hook
    def after_evaluation(self, **kw):
        """
        Executes checkpointing
        :param kw:
        :return:
        """
        # if checkpoint is enabled, save model at each epoch
        if self.checkpoint and not self.param_selection_mode:
            self.model.save(self.trainer.base_path / "checkpoint.pt", checkpoint=True)

    @TrainerPlugin.hook
    def training_interrupt(self, **kw):
        """
        Saves model on interrupt
        :param kw:
        :return:
        """
        if not self.param_selection_mode:
            log.info("Saving model ...")
            self.model.save(self.trainer.base_path / "final-model.pt", checkpoint=self.save_optimizer_state)
            log.info("Done.")

    @TrainerPlugin.hook
    def after_training_loop(self, **kw):
        """
        Saves final model
        :param kw:
        :return:
        """
        # if we do not use dev data for model selection, save final model
        if self.save_final_model and not self.param_selection_mode:
            self.model.save(self.trainer.base_path / "final-model.pt", checkpoint=self.save_optimizer_state)

    @TrainerPlugin.hook
    def best_model(self, **kw):
        """
        Saves best model
        :param kw:
        :return:
        """
        if self.save_best_model:
            log.info("saving best model")
            self.model.save(self.trainer.base_path / "best-model.pt", checkpoint=self.save_optimizer_state)
