from flair.trainers.plugins.base import TrainerPlugin


class AmpPlugin(TrainerPlugin):
    """Simple plugin for AMP."""

    def __init__(self, opt_level) -> None:
        super().__init__()

        self.opt_level = opt_level

        self.wrapped_backward = None

        try:
            from apex import amp

            self.amp = amp
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import apex. Please install apex from "
                "https://www.github.com/nvidia/apex "
                "to enable mixed-precision training."
            ) from exc

    def detach(self, *args, **kwargs):
        # TODO: what does this do?
        super().detach(*args, **kwargs)

        # unwrap trainer backward function
        self.trainer.backward = self.wrapped_backward
        self.wrapped_backward = None

    def backward(self, loss):
        assert self.amp is not None
        optimizer = self.trainer.optimizer

        with self.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    @TrainerPlugin.hook
    def after_setup(self, **kw):
        """Wraps with AMP.

        :param kw:
        :return:
        """
        optimizer = self.trainer.optimizer

        self.trainer.model, self.trainer.optimizer = self.amp.initialize(
            self.model, optimizer, opt_level=self.opt_level
        )

        # replace trainers backward function
        self.wrapped_backward = self.trainer.backward

        self.trainer.backward = self.backward
