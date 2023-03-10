import logging

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import log_line

log = logging.getLogger("flair")


class WandbLogger(TrainerPlugin):
    def __init__(self, *args, use=True, project_name=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.use = use
        self.wandb = None
        self.project_name = project_name

    @TrainerPlugin.hook
    def before_training_setup(self, **kw):
        if not self.use:
            return

        try:
            import wandb

            self.wandb = wandb

            self.wandb.init(project=kw.get("wandb_project", self.project_name))

        except ImportError:
            log_line(log)
            log.warning("ATTENTION! wandb is required for Weight and Biases support!")
            log_line(log)
            self.use = False

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if not self.use:
            return

        if record.is_scalar:
            self.wandb.log({record.name: record.value})
        else:
            raise NotImplementedError

    @TrainerPlugin.hook
    def _training_finally(self, **kw):
        self.writer.close()
