from typing import Optional, Type, cast

from flair.trainers.plugins.base import TrainerPlugin


class SWAPlugin(TrainerPlugin):
    """
    Simple plugin for SWA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = None
        self.SWA: Optional[Type] = None

    @TrainerPlugin.hook
    def before_training_setup(self, learning_rate, **kw):
        """
        initializes SWA and stores learning rate TODO: I think this can be moved
        :param use_swa:
        :param learning_rate:
        :param kw:
        :return:
        """
        self.learning_rate = learning_rate

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
        self.trainer.optimizer = self.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=self.learning_rate)

    @TrainerPlugin.hook
    def after_training_loop(self, **kw):
        """
        Restores SGD weights from SWA
        :param kw:
        :return:
        """
        cast(self.SWA, self.trainer.optimizer).swap_swa_sgd()
