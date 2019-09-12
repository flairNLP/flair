import logging
from pathlib import Path
from typing import List, Union
import time
import sys

import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

try:
    from apex import amp
except ImportError:
    amp = None

import flair
import flair.nn
from flair.data import MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.optim import ExpAnnealLR
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
)

log = logging.getLogger("flair")


class ModelTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        anneal_factor: float = 0.5,
        patience: int = 3,
        min_learning_rate: float = 0.0001,
        eval_mini_batch_size: int = None,
        shuffle: bool = True,
        train_with_dev: bool = False,
        anneal_with_restarts: bool = False,
        sampler=None,
        num_workers: int = 6,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        **kwargs,
    ):
        """
        Trains any class that implements the flair.nn.Model interface.
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param eval_mini_batch_size: Size of mini-batches during evaluation
        :param shuffle: If True, data is shuffled during training
        :param train_with_dev: If True, training is performed using both train+dev data
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param num_workers: Number of workers in your data loader.
        :param use_amp: If True, uses automatic mixed precision optimization (requires APEX to be installed).
        :param kwargs: Other arguments for the Optimizer
        """
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus

        # init optimizer
        self.optimizer_type = optimizer
        self.optimizer: torch.optim.Optimizer = optimizer(
            self.model.parameters(), lr=learning_rate, **kwargs
        )

        # init scheduler
        # minimize training loss if training with dev data, else maximize dev score
        self.train_with_dev: bool = train_with_dev
        anneal_mode = "min" if self.train_with_dev else "max"
        self.scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            self.optimizer,
            factor=anneal_factor,
            patience=patience,
            mode=anneal_mode,
            verbose=True,
        )

        self.epoch: int = 0

        # training parameters
        self.mini_batch_size: int = mini_batch_size
        self.eval_mini_batch_size = (
            mini_batch_size if eval_mini_batch_size is None else eval_mini_batch_size
        )
        self.learning_rate = learning_rate
        self.anneal_factor = anneal_factor
        self.patience: int = patience
        self.anneal_with_restarts: bool = anneal_with_restarts
        self.min_learning_rate: float = min_learning_rate

        # set up training data loader
        train_data = self.corpus.train

        if self.train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        if sampler is not None:
            sampler = sampler(train_data)
            shuffle = False

        self.train_data_loader = DataLoader(
            train_data,
            batch_size=self.mini_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
        )
        self.shuffle = shuffle

        # speed optimizations
        self.use_amp: bool = use_amp
        if self.use_amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=amp_opt_level
            )
        self.num_workers = num_workers

    def train(
        self,
        base_path: Union[Path, str],
        max_epochs: int = 150,
        embeddings_storage_mode: str = "cpu",
        monitor_test: bool = False,
        monitor_train: bool = False,
        use_tensorboard: bool = False,
        checkpoint: bool = False,
        save_final_model: bool = True,
        param_selection_mode: bool = False,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param use_tensorboard: If True, writes out tensorboard information
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :return:
        """

        if type(base_path) == str:
            base_path = Path(base_path)
        weight_extractor = WeightExtractor(base_path)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
                )
                log_line(log)
                use_tensorboard = False
                pass

        if self.use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Learning Parameters:")
        log.info(f' - learning_rate: "{self.learning_rate}"')
        log.info(f' - mini_batch_size: "{self.mini_batch_size}"')
        log.info(f' - patience: "{self.patience}"')
        log.info(f' - anneal_factor: "{self.anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{self.shuffle}"')
        log.info(f' - train_with_dev: "{self.train_with_dev}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = True if not self.train_with_dev else False

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = self.learning_rate

            for self.epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                # get new learning rate
                for group in self.optimizer.param_groups:
                    learning_rate = group["lr"]

                # reload last best model if annealing with restarts is enabled
                if (
                    learning_rate != previous_learning_rate
                    and self.anneal_with_restarts
                    and (base_path / "best-model.pt").exists()
                ):
                    log.info("resetting to best model")
                    self.model.load(base_path / "best-model.pt")

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < self.min_learning_rate:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                train_loss = self._train_one_epoch(
                    self.epoch, embeddings_storage_mode, weight_extractor
                )

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {self.epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}"
                )

                if use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, self.epoch + 1)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        DataLoader(
                            self.corpus.train,
                            batch_size=self.eval_mini_batch_size,
                            num_workers=self.num_workers,
                        ),
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.train, embeddings_storage_mode)

                if log_dev:
                    dev_eval_result, dev_loss = self.model.evaluate(
                        DataLoader(
                            self.corpus.dev,
                            batch_size=self.eval_mini_batch_size,
                            num_workers=self.num_workers,
                        ),
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss)

                    current_score = dev_eval_result.main_score

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)

                    if use_tensorboard:
                        writer.add_scalar("dev_loss", dev_loss, self.epoch + 1)
                        writer.add_scalar(
                            "dev_score", dev_eval_result.main_score, self.epoch + 1
                        )

                if log_test:
                    test_eval_result, test_loss = self.model.evaluate(
                        DataLoader(
                            self.corpus.test,
                            batch_size=self.eval_mini_batch_size,
                            num_workers=self.num_workers,
                        ),
                        base_path / "test.tsv",
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode)

                    if use_tensorboard:
                        writer.add_scalar("test_loss", test_loss, self.epoch + 1)
                        writer.add_scalar(
                            "test_score", test_eval_result.main_score, self.epoch + 1
                        )

                # determine learning rate annealing through scheduler
                self.scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = self.scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in self.optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if new_learning_rate != previous_learning_rate:
                    bad_epochs = self.patience + 1

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # output log file
                with open(loss_txt, "a") as f:

                    # make headers on first epoch
                    if self.epoch == 0:
                        f.write(
                            f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                        )

                        if log_train:
                            f.write(
                                "\tTRAIN_"
                                + "\tTRAIN_".join(
                                    train_eval_result.log_header.split("\t")
                                )
                            )
                        if log_dev:
                            f.write(
                                "\tDEV_LOSS\tDEV_"
                                + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
                            )
                        if log_test:
                            f.write(
                                "\tTEST_LOSS\tTEST_"
                                + "\tTEST_".join(
                                    test_eval_result.log_header.split("\t")
                                )
                            )

                    f.write(
                        f"\n{self.epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )
                    f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.save_checkpoint(base_path / "checkpoint.pt")

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    not self.train_with_dev
                    and not param_selection_mode
                    and current_score == self.scheduler.best
                ):
                    self.model.save(base_path / "best-model.pt")

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if use_tensorboard:
                writer.close()

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test:
            final_score = self.final_test(base_path)
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        if use_tensorboard:
            writer.close()

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

    def save_checkpoint(self, model_file: Union[str, Path]):
        corpus = self.corpus
        self.corpus = None
        torch.save(self, str(model_file), pickle_protocol=4)
        self.corpus = corpus

    @classmethod
    def load_checkpoint(cls, checkpoint, corpus: Corpus):
        model: ModelTrainer = torch.load(checkpoint, map_location=flair.device)
        model.corpus = corpus
        return model

    def _train_one_epoch(
        self, epoch: int, embeddings_storage_mode: str = "none", weight_extractor=None
    ):
        self.model.train()

        train_loss: float = 0

        seen_batches = 0
        total_number_of_batches = len(self.train_data_loader)

        modulo = max(1, int(total_number_of_batches / 10))

        use_amp = False

        # process mini-batches
        batch_time = 0
        for batch_no, batch in enumerate(self.train_data_loader):
            start_time = time.time()
            loss = self.model.forward_loss(batch)

            self.optimizer.zero_grad()
            # Backward
            if use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            seen_batches += 1
            train_loss += loss.item()

            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(batch, embeddings_storage_mode)

            batch_time += time.time() - start_time
            if batch_no % modulo == 0:
                log.info(
                    f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
                    f"{train_loss / seen_batches:.8f} - samples/sec: {self.mini_batch_size * modulo / batch_time:.2f}"
                )
                batch_time = 0
                iteration = epoch * total_number_of_batches + batch_no
                if weight_extractor:
                    weight_extractor.extract_weights(self.model.state_dict(), iteration)

        train_loss /= seen_batches
        return train_loss

    def final_test(self, base_path: Path):

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            DataLoader(
                self.corpus.test,
                batch_size=self.eval_mini_batch_size,
                num_workers=self.num_workers,
            ),
            out_path=base_path / "test.tsv",
            embeddings_storage_mode="none",
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self.model.evaluate(
                    DataLoader(
                        subcorpus.test,
                        batch_size=self.eval_mini_batch_size,
                        num_workers=self.num_workers,
                    ),
                    out_path=base_path / f"{subcorpus.name}-test.tsv",
                    embeddings_storage_mode="none",
                )

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    def find_learning_rate(
        self,
        base_path: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-7,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 32,
        stop_early: bool = True,
        smoothing_factor: float = 0.98,
        **kwargs,
    ) -> Path:
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        # optimizer = self.optimizer(
        #     self.model.parameters(), lr=start_learning_rate, **kwargs
        # )

        train_data = self.corpus.train

        batch_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)

        scheduler = ExpAnnealLR(self.optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        model_device = next(self.model.parameters()).device
        self.model.train()

        for itr, batch in enumerate(batch_loader):
            loss = self.model.forward_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            scheduler.step(1)
            learning_rate = scheduler.get_lr()[0]

            loss_item = loss.item()
            if itr == 0:
                best_loss = loss_item
            else:
                if smoothing_factor > 0:
                    moving_avg_loss = (
                        smoothing_factor * moving_avg_loss
                        + (1 - smoothing_factor) * loss_item
                    )
                    loss_item = moving_avg_loss / (1 - smoothing_factor ** (itr + 1))
                if loss_item < best_loss:
                    best_loss = loss

            if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                log_line(log)
                log.info("loss diverged - stopping early!")
                break

            if itr > iterations:
                break

            with open(str(learning_rate_tsv), "a") as f:
                f.write(
                    f"{itr}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
                )

        self.model.load_state_dict(model_state)
        self.model.to(model_device)

        log_line(log)
        log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
        log_line(log)

        return Path(learning_rate_tsv)
