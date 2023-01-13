from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import WeightExtractor


class WeightExtractorPlugin(TrainerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight_extractor = None

    @TrainerPlugin.hook
    def before_training_setup(self, param_selection_mode, write_weights, base_path, **kw):
        if not param_selection_mode and write_weights:
            self.weight_extractor = WeightExtractor(base_path)

    @TrainerPlugin.hook
    def after_training_batch(self, batch_no, epoch, total_number_of_batches, **kw):
        modulo = max(1, int(total_number_of_batches / 10))
        iteration = epoch * total_number_of_batches + batch_no

        if (iteration + 1) % modulo == 0 and self.weight_extractor is not None:
            self.weight_extractor.extract_weights(self.model.state_dict(), iteration)
