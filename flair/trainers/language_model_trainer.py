import time, datetime
import random
import sys
from pathlib import Path
from typing import Union

from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.optim.sgd import SGD

try:
    from apex import amp
except ImportError:
    amp = None

import flair
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.optim import *
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class TextDataset(Dataset):
    def __init__(
        self,
        path: Path,
        dictionary: Dictionary,
        expand_vocab: bool = False,
        forward: bool = True,
        split_on_char: bool = True,
        random_case_flip: bool = True,
        document_delimiter: str = '\n',
        shuffle: bool = True,
    ):

        assert path.exists()

        self.files = None
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

    def __getitem__(self, index=0) -> torch.tensor:
        return self.charsplit(
            self.files[index],
            self.expand_vocab,
            self.forward,
            self.split_on_char,
            self.random_case_flip,
        )

    def charsplit(
        self,
        path: Path,
        expand_vocab=False,
        forward=True,
        split_on_char=True,
        random_case_flip=True,
    ) -> torch.tensor:

        """Tokenizes a text file on character basis."""
        assert path.exists()

        lines = [doc + self.document_delimiter
                 for doc in open(path, "r", encoding="utf-8").read().split(self.document_delimiter) if doc]

        log.info(f"read text file with {len(lines)} lines")
        if self.shuffle:
            random.shuffle(lines)
            log.info(f"shuffled")

        tokens = 0
        for line in lines:

            if split_on_char:
                chars = list(line)
            else:
                chars = line.split()

            tokens += len(chars)

            # Add chars to the dictionary
            if expand_vocab:
                for char in chars:
                    self.dictionary.add_item(char)

        ids = torch.zeros(tokens, dtype=torch.long)
        if forward:
            # charsplit file content
            token = 0
            for line in lines:
                if random_case_flip:
                    line = self.random_casechange(line)

                if split_on_char:
                    chars = list(line)
                else:
                    chars = line.split()

                for char in chars:
                    if token >= tokens:
                        break
                    ids[token] = self.dictionary.get_idx_for_item(char)
                    token += 1
        else:
            # charsplit file content
            token = tokens - 1
            for line in lines:
                if random_case_flip:
                    line = self.random_casechange(line)

                if split_on_char:
                    chars = list(line)
                else:
                    chars = line.split()

                for char in chars:
                    if token >= tokens:
                        break
                    ids[token] = self.dictionary.get_idx_for_item(char)
                    token -= 1

        return ids

    @staticmethod
    def random_casechange(line: str) -> str:
        no = random.randint(0, 99)
        if no is 0:
            line = line.lower()
        if no is 1:
            line = line.upper()
        return line

    def tokenize(self, path: Path):
        """Tokenizes a text file."""
        assert path.exists()
        # Add words to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r") as f:
            ids = torch.zeros(tokens, dtype=torch.long, device=flair.device)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class TextCorpus(object):
    def __init__(
        self,
        path: Union[Path, str],
        dictionary: Dictionary,
        forward: bool = True,
        character_level: bool = True,
        random_case_flip: bool = True,
        document_delimiter: str = '\n',
    ):
        self.dictionary: Dictionary = dictionary
        self.forward = forward
        self.split_on_char = character_level
        self.random_case_flip = random_case_flip
        self.document_delimiter: str = document_delimiter

        if type(path) == str:
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

        # TextDataset returns a list. valid and test are only one file, so return the first element
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
        optimizer: Optimizer = SGD,
        test_mode: bool = False,
        epoch: int = 0,
        split: int = 0,
        loss: float = 10000,
        optimizer_state: dict = None,
    ):
        self.model: LanguageModel = model
        self.optimizer: Optimizer = optimizer
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
                    "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        add_file_handler(log, base_path / "training.log")

        number_of_splits: int = len(self.corpus.train)

        val_data = self._batchify(self.corpus.valid, mini_batch_size)

        # error message if the validation dataset is too small
        if val_data.size(0) == 1:
            raise RuntimeError(
                f"ERROR: Your validation dataset is too small. For your mini_batch_size, the data needs to "
                f"consist of at least {mini_batch_size * 2} characters!"
            )

        base_path.mkdir(parents=True, exist_ok=True)
        loss_txt = base_path / "loss.txt"
        savefile = base_path / "best-lm.pt"

        try:
            best_val_loss = self.loss
            optimizer = self.optimizer(
                self.model.parameters(), lr=learning_rate, **kwargs
            )
            if self.optimizer_state is not None:
                optimizer.load_state_dict(self.optimizer_state)

            if isinstance(optimizer, (AdamW, SGDW)):
                scheduler: ReduceLRWDOnPlateau = ReduceLRWDOnPlateau(
                    optimizer, verbose=True, factor=anneal_factor, patience=patience
                )
            else:
                scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
                    optimizer, verbose=True, factor=anneal_factor, patience=patience
                )

            if use_amp:
                self.model, optimizer = amp.initialize(
                    self.model, optimizer, opt_level=amp_opt_level
                )

            training_generator = DataLoader(
                self.corpus.train, shuffle=False, num_workers=num_workers
            )

            for epoch in range(self.epoch, max_epochs):
                epoch_start_time = time.time()
                # Shuffle training files randomly after serially iterating through corpus one
                if epoch > 0:
                    training_generator = DataLoader(
                        self.corpus.train, shuffle=True, num_workers=num_workers
                    )
                    self.model.save_checkpoint(
                        base_path / f"epoch_{epoch}.pt",
                        optimizer,
                        epoch,
                        0,
                        best_val_loss,
                    )

                # iterate through training data, starting at self.split (for checkpointing)
                for curr_split, train_slice in enumerate(
                    training_generator, self.split
                ):

                    if sequence_length < grow_to_sequence_length:
                        sequence_length += 1
                    log.info(f"Sequence length is {sequence_length}")

                    split_start_time = time.time()
                    # off by one for printing
                    curr_split += 1
                    train_data = self._batchify(train_slice.flatten(), mini_batch_size)

                    log.info(
                        "Split %d" % curr_split
                        + "\t - ({:%H:%M:%S})".format(datetime.datetime.now())
                    )

                    for group in optimizer.param_groups:
                        learning_rate = group["lr"]

                    # go into train mode
                    self.model.train()

                    # reset variables
                    hidden = self.model.init_hidden(mini_batch_size)

                    # not really sure what this does
                    ntokens = len(self.corpus.dictionary)

                    total_loss = 0
                    start_time = time.time()

                    for batch, i in enumerate(
                        range(0, train_data.size(0) - 1, sequence_length)
                    ):
                        data, targets = self._get_batch(train_data, i, sequence_length)

                        if not data.is_cuda and cuda.is_available():
                            log.info(
                                "Batch %d is not on CUDA, training will be very slow"
                                % (batch)
                            )
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

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                        optimizer.step()

                        total_loss += loss.data

                        # We detach the hidden state from how it was previously produced.
                        # If we didn't, the model would try backpropagating all the way to start of the dataset.
                        hidden = self._repackage_hidden(hidden)

                        # explicitly remove loss to clear up memory
                        del loss, output, rnn_output

                        if batch % self.log_interval == 0 and batch > 0:
                            cur_loss = total_loss.item() / self.log_interval
                            elapsed = time.time() - start_time
                            log.info(
                                "| split {:3d} /{:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                                "loss {:5.2f} | ppl {:8.2f}".format(
                                    curr_split,
                                    number_of_splits,
                                    batch,
                                    len(train_data) // sequence_length,
                                    elapsed * 1000 / self.log_interval,
                                    cur_loss,
                                    math.exp(cur_loss),
                                )
                            )
                            total_loss = 0
                            start_time = time.time()

                    log.info(
                        "%d seconds for train split %d"
                        % (time.time() - split_start_time, curr_split)
                    )

                    ###############################################################################
                    self.model.eval()

                    val_loss = self.evaluate(val_data, mini_batch_size, sequence_length)
                    scheduler.step(val_loss)

                    log.info("best loss so far {:5.2f}".format(best_val_loss))

                    log.info(self.model.generate_text())

                    if checkpoint:
                        self.model.save_checkpoint(
                            base_path / "checkpoint.pt",
                            optimizer,
                            epoch,
                            curr_split,
                            best_val_loss,
                        )

                    # Save the model if the validation loss is the best we've seen so far.
                    if val_loss < best_val_loss:
                        self.model.best_score = best_val_loss
                        self.model.save(savefile)
                        best_val_loss = val_loss

                    ###############################################################################
                    # print info
                    ###############################################################################
                    log.info("-" * 89)

                    summary = (
                        "| end of split {:3d} /{:3d} | epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                        "valid ppl {:8.2f} | learning rate {:3.4f}".format(
                            curr_split,
                            number_of_splits,
                            epoch + 1,
                            (time.time() - split_start_time),
                            val_loss,
                            math.exp(val_loss),
                            learning_rate,
                        )
                    )

                    with open(loss_txt, "a") as myfile:
                        myfile.write("%s\n" % summary)

                    log.info(summary)
                    log.info("-" * 89)

                log.info("Epoch time: %.2f" % (time.time() - epoch_start_time))

        except KeyboardInterrupt:
            log.info("-" * 89)
            log.info("Exiting from training early")

        ###############################################################################
        # final testing
        ###############################################################################
        test_data = self._batchify(self.corpus.test, mini_batch_size)
        test_loss = self.evaluate(test_data, mini_batch_size, sequence_length)

        summary = "TEST: valid loss {:5.2f} | valid ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
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

        data = source[i : i + seq_len].clone().detach()
        target = source[i + 1 : i + 1 + seq_len].view(-1).clone().detach()

        data = data.to(flair.device)
        target = target.to(flair.device)

        return data, target

    @staticmethod
    def _repackage_hidden(h):
        """Wraps hidden states in new tensors, to detach them from their history."""
        return tuple(v.clone().detach() for v in h)

    @staticmethod
    def load_from_checkpoint(
        checkpoint_file: Path, corpus: TextCorpus, optimizer: Optimizer = SGD
    ):
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
