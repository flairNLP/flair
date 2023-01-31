import sys

from flair.trainers.plugins.base import TrainerPlugin

try:
    from apex import amp
except ImportError:
    amp = None


class AmpPlugin(TrainerPlugin):
    def __init__(self):
        super().__init__()
        self.use = None
        self.opt_level = None
        self.wrapped_backward = None

    @TrainerPlugin.hook
    def before_training_setup(self, use_amp, amp_opt_level, **kw):
        self.use = use_amp
        self.opt_level = amp_opt_level

        if self.use:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from "
                    "https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

    def detach(self, *args, **kwargs):
        super().detach(*args, **kwargs)

        self.trainer.backward = self.wrapped_backward
        self.wrapped_backward = None

    def backward(self, loss):
        optimizer = self.trainer.optimizer

        if self.use:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

    @TrainerPlugin.hook
    def after_optimizer_setup(self, **kw):
        optimizer = self.trainer.optimizer

        if self.use:
            self.trainer.model, self.trainer.optimizer = amp.initialize(self.model, optimizer, opt_level=self.opt_level)

            # replace trainers backward function
            self.wrapped_backward = self.trainer.backward

            self.trainer.backward = self.backward
