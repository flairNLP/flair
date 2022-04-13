import datetime
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Type, Union

import torch
from torch import cuda
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from flair.optim import SGDW, ReduceLRWDOnPlateau

try:
    from apex import amp
except ImportError:
    amp = None

import flair
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class TextDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        dictionary: Dictionary,
        expand_vocab: bool = False,
        forward: bool = True,
        split_on_char: bool = True,
        random_case_flip: bool = True,
        document_delimiter: str = "\n",
        shuffle: bool = True,
    ):
        path = Path(path)
        assert path.exists()

        self.path = path
        self.dictionary = dictionary
        self.split_on_char = split_on_char
        self.forward = forward
        self.random_case_flip = random_case_flip
        self.expand_vocab = expand_vocab
        self.document_delimiter = document_delimiter
        self.shuffle = shuffle

        if path.is_dir():
            self.files = sorted([f for f in path.iterdir() if f.exists()])
        else:
            self.files = [path]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index=0) -> torch.Tensor:
        """Tokenizes a text file on character basis."""
        if type(self.files[index]) is str:
            self.files[index] = Path(self.files[index])
        assert self.files[index].exists()

        with self.files[index].open("r", encoding="utf-8") as fin:
            text_lines: Iterable[str] = (
                doc + self.document_delimiter for doc in fin.read().split(self.document_delimiter) if doc
            )
            if self.random_case_flip:
                text_lines = map(self.random_casechange, text_lines)
            lines = list(map(list if self.split_on_char else str.split, text_lines))  # type: ignore # noqa: E501

        log.info(f"read text file with {len(lines)} lines")

        if self.shuffle:
            random.shuffle(lines)
            log.info("shuffled")

        if self.expand_vocab:
            for chars in lines:
                for char in chars:
                    self.dictionary.add_item(char)

        ids = torch.tensor(
            [self.dictionary.get_idx_for_item(char) for chars in lines for char in chars],
            dtype=torch.long,
        )
        if not self.forward:
            ids = ids.flip(0)
        return ids

    @staticmethod
    def random_casechange(line: str) -> str:
        no = random.randint(0, 99)
        if no == 0:
            line = line.lower()
        if no == 1:
            line = line.upper()
        return line


class TextCorpus(object):
    def __init__(
        self,
        path: Union[Path, str],
        dictionary: Dictionary,
        forward: bool = True,
        character_level: bool = True,
        random_case_flip: bool = True,
        document_delimiter: str = "\n",
    ):
        self.dictionary: Dictionary = dictionary
        self.forward = forward
        self.split_on_char = character_level
        self.random_case_flip = random_case_flip
        self.document_delimiter: str = document_delimiter

        path = Path(path)

        self.train = TextDataset(
            path / "train",
            dictionary,
            False,
            self.forward,
            self.split_on_char,
            self.random_case_flip,
            document_delimiter=self.document_delimiter,
            shuffle=True,
        )

        # TextDataset returns a list. valid and test are only one file,
        # so return the first element
        self.valid = TextDataset(
            path / "valid.txt",
            dictionary,
            False,
            self.forward,
            self.split_on_char,
            self.random_case_flip,
            document_delimiter=document_delimiter,
            shuffle=False,
        )[0]
        self.test = TextDataset(
            path / "test.txt",
            dictionary,
            False,
            self.forward,
            self.split_on_char,
            self.random_case_flip,
            document_delimiter=document_delimiter,
            shuffle=False,
        )[0]


class LanguageModelTrainer:
    def __init__(
        self,
        model: LanguageModel,
        corpus: TextCorpus,
        optimizer: Type[Optimizer] = SGD,
        test_mode: bool = False,
        epoch: int = 0,
        split: int = 0,
        loss: float = 10000,
        optimizer_state: dict = None,
    ):
        self.model: LanguageModel = model
        self.optimizer: Type[Optimizer] = optimizer
        self.corpus: TextCorpus = corpus
        self.test_mode: bool = test_mode

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.log_interval = 100
        self.epoch = epoch
        self.split = split
        self.loss = loss
        self.optimizer_state = optimizer_state

    def train(
        self,
        base_path: Union[Path, str],
        sequence_length: int,
        learning_rate: float = 20,
        mini_batch_size: int = 100,
        anneal_factor: float = 0.25,
        patience: int = 10,
        clip=0.25,
        max_epochs: int = 1000,
        checkpoint: bool = False,
        grow_to_sequence_length: int = 0,
        num_workers: int = 2,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        **kwargs,
    ):

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from "
                    "https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        # cast string to Path
        base_path = Path(base_path)

        number_of_splits: int = len(self.corpus.train)

        val_data = self._batchify(self.corpus.valid, mini_batch_size)

        # error message if the validation dataset is too small
        if val_data.size(0) == 1:
            raise RuntimeError(
                f"ERROR: Your validation dataset is too small. For your "
                f"mini_batch_size, the data needs to "
                f"consist of at least {mini_batch_size * 2} characters!"
            )

        base_path.mkdir(parents=True, exist_ok=True)
        loss_txt = base_path / "loss.txt"
        savefile = base_path / "best-lm.pt"

        try:
            log_handler = add_file_handler(log, base_path / "training.log")

            best_val_loss = self.loss
            kwargs["lr"] = learning_rate
            optimizer = self.optimizer(self.model.parameters(), **kwargs)
            if self.optimizer_state is not None:
                optimizer.load_state_dict(self.optimizer_state)

            if isinstance(optimizer, (AdamW, SGDW)):
                scheduler: ReduceLROnPlateau = ReduceLRWDOnPlateau(
                    optimizer, verbose=True, factor=anneal_factor, patience=patience
                )
            else:
                scheduler = ReduceLROnPlateau(optimizer, verbose=True, factor=anneal_factor, patience=patience)

            if use_amp:
                self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=amp_opt_level)

            training_generator = DataLoader(self.corpus.train, shuffle=False, num_workers=num_workers)

            for epoch in range(self.epoch, max_epochs):
                epoch_start_time = time.time()
                # Shuffle training files randomly after serially iterating
                # through corpus one
                if epoch > 0:
                    training_generator = DataLoader(self.corpus.train, shuffle=True, num_workers=num_workers)
                    self.model.save_checkpoint(
                        base_path / f"epoch_{epoch}.pt",
                        optimizer,
                        epoch,
                        0,
                        best_val_loss,
                    )

                # iterate through training data, starting at
                # self.split (for checkpointing)
                for curr_split, train_slice in enumerate(training_generator, self.split):

                    if sequence_length < grow_to_sequence_length:
                        sequence_length += 1
                    log.info(f"Sequence length is {sequence_length}")

                    split_start_time = time.time()
                    # off by one for printing
                    curr_split += 1
                    train_data = self._batchify(train_slice.flatten(), mini_batch_size)

                    log.info("Split %d" % curr_split + "\t - ({:%H:%M:%S})".format(datetime.datetime.now()))

                    for group in optimizer.param_groups:
                        learning_rate = group["lr"]

                    # go into train mode
                    self.model.train()

                    # reset variables
                    hidden = self.model.init_hidden(mini_batch_size)

                    # not really sure what this does
                    ntokens = len(self.corpus.dictionary)

                    total_loss = torch.zeros(1, device=flair.device)
                    start_time = time.time()

                    for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):
                        data, targets = self._get_batch(train_data, i, sequence_length)

                        if not data.is_cuda and cuda.is_available():
                            log.info("Batch %d is not on CUDA, training will be very slow" % (batch))
                            raise Exception("data isnt on cuda")

                        self.model.zero_grad()
                        optimizer.zero_grad()

                        # do the forward pass in the model
                        output, rnn_output, hidden = self.model.forward(data, hidden)

                        # try to predict the targets
                        loss = self.loss_function(output.view(-1, ntokens), targets)
                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                        optimizer.step()

                        total_loss += loss.data

                        # We detach the hidden state from how it was
                        # previously produced.
                        # If we didn't, the model would try backpropagating
                        # all the way to start of the dataset.
                        hidden = self._repackage_hidden(hidden)

                        # explicitly remove loss to clear up memory
                        del loss, output, rnn_output

                        if batch % self.log_interval == 0 and batch > 0:
                            cur_loss = total_loss.item() / self.log_interval
                            elapsed = time.time() - start_time
                            log.info(
                                f"| split {curr_split:3d}/{number_of_splits:3d} | {batch:5d}/{len(train_data) // sequence_length:5d} batches "
                                f"| ms/batch {elapsed * 1000 / self.log_interval:5.2f} | loss {cur_loss:5.4f} | ppl {math.exp(cur_loss):5.4f}"
                            )
                            total_loss = torch.zeros(1, device=flair.device)
                            start_time = time.time()

                    ##########################################################
                    self.model.eval()

                    val_loss = self.evaluate(val_data, mini_batch_size, sequence_length)

                    # Save the model if the validation loss is the best we've
                    # seen so far.
                    if val_loss < best_val_loss:
                        self.model.save(savefile)
                        best_val_loss = val_loss
                        log.info("best split so far")

                    scheduler.step(val_loss)

                    log.info(f"best loss so far {best_val_loss:5.8f}")

                    log.info(self.model.generate_text())

                    if checkpoint:
                        self.model.save_checkpoint(
                            base_path / "checkpoint.pt",
                            optimizer,
                            epoch,
                            curr_split,
                            best_val_loss,
                        )

                    ##########################################################
                    # print info
                    ##########################################################
                    log.info("-" * 89)

                    summary = (
                        f"| end of split {curr_split:3d} /{number_of_splits:3d} | epoch {epoch + 1:3d} | time: "
                        f"{(time.time() - split_start_time):5.2f}s | valid loss {val_loss:5.4f} | valid ppl "
                        f"{math.exp(val_loss):5.4f} | learning rate {learning_rate:3.4f}"
                    )

                    with open(loss_txt, "a") as myfile:
                        myfile.write("%s\n" % summary)

                    log.info(summary)
                    log.info("-" * 89)
                    log.info("%d seconds for train split %d" % (time.time() - split_start_time, curr_split))

                log.info("Epoch time: %.2f" % (time.time() - epoch_start_time))

        except KeyboardInterrupt:
            log.info("-" * 89)
            log.info("Exiting from training early")
        finally:
            if log_handler is not None:
                log_handler.close()
                log.removeHandler(log_handler)

        ###############################################################################
        # final testing
        ###############################################################################
        test_data = self._batchify(self.corpus.test, mini_batch_size)
        test_loss = self.evaluate(test_data, mini_batch_size, sequence_length)

        summary = f"TEST: valid loss {test_loss:5.4f} | valid ppl {math.exp(test_loss):8.4f}"
        with open(loss_txt, "a") as myfile:
            myfile.write("%s\n" % summary)

        log.info(summary)
        log.info("-" * 89)

    def evaluate(self, data_source, eval_batch_size, sequence_length):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            ntokens = len(self.corpus.dictionary)

            hidden = self.model.init_hidden(eval_batch_size)

            for i in range(0, data_source.size(0) - 1, sequence_length):
                data, targets = self._get_batch(data_source, i, sequence_length)
                prediction, rnn_output, hidden = self.model.forward(data, hidden)
                output_flat = prediction.view(-1, ntokens)
                total_loss += len(data) * self.loss_function(output_flat, targets).data
                hidden = self._repackage_hidden(hidden)
            return total_loss.item() / len(data_source)

    @staticmethod
    def _batchify(data, batch_size):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data

    @staticmethod
    def _get_batch(source, i, sequence_length):
        seq_len = min(sequence_length, len(source) - 1 - i)

        data = source[i : i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].view(-1)

        data = data.to(flair.device)
        target = target.to(flair.device)

        return data, target

    @staticmethod
    def _repackage_hidden(h):
        """Wraps hidden states in new tensors, to detach them from their history."""
        return tuple(v.detach() for v in h)

    @staticmethod
    def load_checkpoint(
        checkpoint_file: Union[str, Path],
        corpus: TextCorpus,
        optimizer: Type[Optimizer] = SGD,
    ):
        if type(checkpoint_file) is str:
            checkpoint_file = Path(checkpoint_file)

        checkpoint = LanguageModel.load_checkpoint(checkpoint_file)
        return LanguageModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            split=checkpoint["split"],
            loss=checkpoint["loss"],
            optimizer_state=checkpoint["optimizer_state_dict"],
        )
