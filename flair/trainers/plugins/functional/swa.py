from typing import Optional, Type, cast

from flair.trainers.plugins.base import TrainerPlugin


class SWAPlugin(TrainerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_swa = None
        self.learning_rate = None
        self.SWA: Optional[Type] = None

    @TrainerPlugin.hook
    def before_training_setup(self, use_swa, learning_rate, **kw):
        self.use_swa = use_swa
        self.learning_rate = learning_rate

        if self.use_swa:
            import torchcontrib

            self.SWA = torchcontrib.optim.SWA

    @TrainerPlugin.hook
    def after_optimizer_setup(self, optimizer, **kw):
        if self.use_swa:
            optimizer = self.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=self.learning_rate)

    @TrainerPlugin.hook
    def after_training_loop(self, **kw):
        if self.use_swa:
            cast(self.SWA, self.trainer.optimizer).swap_swa_sgd()
