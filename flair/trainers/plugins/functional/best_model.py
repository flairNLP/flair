from flair.trainers.plugins.base import TrainerPlugin


class BestModelPlugin(TrainerPlugin):
    """Assesses whether the current model is the best model seen so far.

    If (after having trained for a epoch) the current model has the best
    validations score, the `best_model` event is dispatched.
    """

    provided_events = {"best_model"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.minimize_objective: bool = None

        self.primary_metric: str = None
        self.auxiliary_metric: str = None

        self.best_value: float = None
        self.primary_value: float = None
        self.auxiliary_value: float = None

    @TrainerPlugin.hook
    def before_training_setup(
        self,
        train_with_dev,
        anneal_against_dev_loss,
        **kw,
    ):
        # minimize training loss if training with dev data, else maximize dev score
        self.minimize_objective = train_with_dev or anneal_against_dev_loss
        self.best_validation_score = 100000000000 if self.minimize_objective else -1.0

        if train_with_dev:
            self.primary_metric = "train/loss"
        elif anneal_against_dev_loss:
            self.primary_metric = "dev/loss"
        else:
            self.primary_metric = "dev/score"
            self.auxiliary_metric = "dev/loss"

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        # record relevant values for scheduling
        if record.name == self.primary_metric:
            self.primary_value = record.value

        if record.name == self.auxiliary_metric:
            self.auxiliary_value = record.value

    @TrainerPlugin.hook
    def after_evaluation(self, epoch, **kw):
        assert self.primary_value is not None

        # determine if this is the best model
        if (self.minimize_objective and self.primary_value < self.best_value) or (
            not self.minimize_objective and self.primary_value > self.best_value
        ):
            # new best validation score
            self.best_value = self.primary_value

            self.trainer.dispatch(
                "best_model", primary_value=self.primary_value, auxiliary_value=self.auxiliary_value, epoch=epoch
            )

        self.primary_value = None
        self.auxiliary_value = None
