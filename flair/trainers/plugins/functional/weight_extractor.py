from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import WeightExtractor


class WeightExtractorPlugin(TrainerPlugin):
    """Simple Plugin for weight extraction."""

    def __init__(self, base_path) -> None:
        super().__init__()
        self.weight_extractor = WeightExtractor(base_path)

    @TrainerPlugin.hook
    def after_training_batch(self, batch_no, epoch, total_number_of_batches, **kw):
        """Extracts weights.

        :param batch_no:
        :param epoch:
        :param total_number_of_batches:
        :param kw:
        :return:
        """
        modulo = max(1, int(total_number_of_batches / 10))
        iteration = epoch * total_number_of_batches + batch_no

        if (iteration + 1) % modulo == 0:
            self.weight_extractor.extract_weights(self.model.state_dict(), iteration)
