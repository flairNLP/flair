import logging
from time import time

import flair
from flair.train_util import log_line
from flair.trainers.plugins.base import TrainerPlugin

log = logging.getLogger("flair")


class LoggingPlugin(TrainerPlugin):
    def __init__(self, log_modulo=None):
        super().__init__()

        self.log_modulo = log_modulo
        self.cycle_momentum = None

        self.average_over = None
        self.total_loss = None

        self.lr_info = None
        self.momentum_info = None

    @TrainerPlugin.hook
    def before_training_loop(
        self,
        patience,
        anneal_factor,
        max_epochs,
        shuffle,
        train_with_dev,
        batch_growth_annealing,
        embeddings_storage_mode,
        cycle_momentum,
        **kw,
    ):
        optimizer = self.trainer.optimizer
        lr_info = ",".join([f"{group['lr']:.6f}" for group in optimizer.param_groups])

        log_line(log)
        log.info(f'Model: "{self.trainer.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.trainer.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{lr_info}"')
        log.info(f' - mini_batch_size: "{self.trainer.mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log.info(f' - batch_growth_annealing: "{batch_growth_annealing}"')
        log_line(log)
        log.info(f'Model training base path: "{self.trainer.base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

        self.cycle_momentum = cycle_momentum
        self.momentum_info = ""
        self.lr_info = ""

    @TrainerPlugin.hook
    def before_training_epoch(self, **kw):
        log_line(log)

        self.average_over = 0
        self.total_loss = 0

        self.start_time = time.time()

    @TrainerPlugin.hook
    def after_training_batch_step(self, loss, datapoint_count):
        self.total_loss += loss
        self.average_over += datapoint_count

    @TrainerPlugin.hook
    def metric_recorded(self, record):
        if record.name == "learning_rate":
            if record.is_scalar:
                self.lr_info = f" - lr: {record.value:.4f}"
            else:
                self.lr_info = " - lr: " + ",".join([f"{m:.4f}" for m in record.value])

        elif record.name == "momentum" and self.cycle_momentum:
            if record.is_scalar:
                self.momentum_info = f" - momentum: {record.value:.4f}"
            else:
                self.momentum_info = " - momentum: " + ",".join([f"{m:.4f}" for m in record.value])

    @TrainerPlugin.hook
    def after_training_batch(self, batch_no, epoch, total_number_of_batches, **kw):
        modulo = self.log_modulo

        if modulo is None:
            modulo = max(1, int(total_number_of_batches / 10))

        if (batch_no + 1) % modulo == 0:
            intermittent_loss = (
                self.total_loss / self.average_over if self.average_over > 0 else self.train_loss / (batch_no + 1)
            )

            end_time = time.time()

            log.info(
                f"epoch {epoch}"
                f" - iter {batch_no + 1}/{total_number_of_batches}"
                f" - loss {intermittent_loss:.8f}"
                f" - time (sec): {(end_time - self.start_time):.2f}"
                f" - samples/sec: {self.average_over / (end_time - self.start_time):.2f}"
                f" - {self.lr_info}{self.momentum_info}"
            )

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch, **kw):
        log_line(log)
        log.info(f"EPOCH {epoch} done: loss {self.total_loss:.4f}{self.lr_info}")
