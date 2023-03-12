from typing import Optional, Type, cast

from flair.trainers.plugins.base import TrainerPlugin


class SWAPlugin(TrainerPlugin):
    """
    Simple plugin for SWA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_swa = None
        self.learning_rate = None
        self.SWA: Optional[Type] = None

    @TrainerPlugin.hook
    def before_training_setup(self, use_swa, learning_rate, **kw):
        """
        initializes SWA and stores learning rate TODO: I think this can be moved
        :param use_swa:
        :param learning_rate:
        :param kw:
        :return:
        """
        self.use_swa = use_swa # TODO: is this var necessary?
        self.learning_rate = learning_rate

        if self.use_swa: ## TODO: are these ifs still necessary?

            import torchcontrib

            self.SWA = torchcontrib.optim.SWA

    @TrainerPlugin.hook
    def after_optimizer_setup(self, optimizer, **kw):
        """
        wraps the optimizer with SWA
        :param optimizer:
        :param kw:
        :return:
        """
        if self.use_swa: ## TODO: are these ifs still necessary?
            optimizer = self.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=self.learning_rate)

    @TrainerPlugin.hook
    def after_training_loop(self, **kw):
        """
        Restores SGD weights from SWA
        :param kw:
        :return:
        """
        if self.use_swa: ## TODO: are these ifs still necessary?
            cast(self.SWA, self.trainer.optimizer).swap_swa_sgd()
