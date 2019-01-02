import time, datetime
import random
import logging
from pathlib import Path
from typing import Union

from torch.autograd import Variable
from torch.optim.sgd import SGD

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.optim import *
from flair.training_utils import add_file_handler


log = logging.getLogger('flair')


class TextCorpus(object):
    def __init__(self, path: Path, dictionary: Dictionary, forward: bool = True, character_level: bool = True):

        self.forward = forward
        self.split_on_char = character_level
        self.train_path = path / 'train'

        self.train_files = sorted(
            [f for f in self.train_path.iterdir() if f.exists()])

        self.dictionary: Dictionary = dictionary

        self.current_train_file_index = len(self.train_files)

        self.valid = self.charsplit(path / 'valid.txt',
                                    forward=forward,
                                    split_on_char=self.split_on_char)

        self.test = self.charsplit(path / 'test.txt',
                                   forward=forward,
                                   split_on_char=self.split_on_char)

    @property
    def is_last_slice(self) -> bool:
        if self.current_train_file_index >= len(self.train_files) - 1:
            return True
        return False

    def get_next_train_slice(self):

        self.current_train_file_index += 1

        if self.current_train_file_index >= len(self.train_files):
            self.current_train_file_index = 0
            random.shuffle(self.train_files)

        current_train_file = self.train_files[self.current_train_file_index]

        train_slice = self.charsplit(current_train_file,
                                    expand_vocab=False,
                                    forward=self.forward,
                                    split_on_char=self.split_on_char)

        return train_slice

    def charsplit(self, path: Path, expand_vocab=False, forward=True, split_on_char=True) -> torch.LongTensor:

        """Tokenizes a text file on character basis."""
        assert path.exists()

        #
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f:

                if split_on_char:
                    chars = list(line)
                else:
                    chars = line.split()

                tokens += len(chars)

                # Add chars to the dictionary
                if expand_vocab:
                    for char in chars:
                        self.dictionary.add_item(char)

        if forward:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
                        ids[token] = self.dictionary.get_idx_for_item(char)
                        token += 1
        else:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = tokens - 1
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
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
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class LanguageModelTrainer:

    def __init__(self,
                 model: LanguageModel,
                 corpus: TextCorpus,
                 optimizer: Optimizer = SGD,
                 test_mode: bool = False,
                 epoch: int = 0,
                 split: int = 0,
                 loss: float = 10000,
                 optimizer_state: dict = None
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

    def train(self,
              base_path: Union[Path, str],
              sequence_length: int,
              learning_rate: float = 20,
              mini_batch_size: int = 100,
              anneal_factor: float = 0.25,
              patience: int = 10,
              clip=0.25,
              max_epochs: int = 1000,
              checkpoint: bool = False,
              **kwargs):

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        add_file_handler(log, base_path / 'training.log')

        number_of_splits: int = len(self.corpus.train_files)

        # an epoch has a number, so calculate total max splits bby multiplying max_epochs with number_of_splits
        max_splits: int = number_of_splits * max_epochs

        val_data = self._batchify(self.corpus.valid, mini_batch_size)

        base_path.mkdir(parents=True, exist_ok=True)
        loss_txt = base_path / 'loss.txt'
        savefile = base_path / 'best-lm.pt'

        try:

            epoch = self.epoch
            best_val_loss = self.loss
            optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
            if self.optimizer_state is not None:
                optimizer.load_state_dict(self.optimizer_state)

            if isinstance(optimizer, (AdamW, SGDW)):
                scheduler: ReduceLRWDOnPlateau = ReduceLRWDOnPlateau(optimizer, verbose=True,
                                                                     factor=anneal_factor,
                                                                     patience=patience)
            else:
                scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, verbose=True,
                                                                 factor=anneal_factor,
                                                                 patience=patience)

            train_data = None
            for split in range(1 + self.split, max_splits + 1):

                # after pass over all splits, increment epoch count
                if (split - 1) % number_of_splits == 0:
                    epoch += 1

                log.info('Split %d' % split + '\t - ({:%H:%M:%S})'.format(datetime.datetime.now()))

                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                if train_data is None or len(self.corpus.train_files) > 1:
                    train_slice = self.corpus.get_next_train_slice()
                    train_data = self._batchify(train_slice, mini_batch_size)

                log.info('\t({:%H:%M:%S})'.format(datetime.datetime.now()))

                # go into train mode
                self.model.train()

                # reset variables
                epoch_start_time = time.time()
                total_loss = 0
                start_time = time.time()

                hidden = self.model.init_hidden(mini_batch_size)

                # not really sure what this does
                ntokens = len(self.corpus.dictionary)

                # do batches
                for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):

                    data, targets = self._get_batch(train_data, i, sequence_length)

                    # Starting each batch, we detach the hidden state from how it was previously produced.
                    # If we didn't, the model would try backpropagating all the way to start of the dataset.
                    hidden = self._repackage_hidden(hidden)

                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # do the forward pass in the model
                    output, rnn_output, hidden = self.model.forward(data, hidden)

                    # try to predict the targets
                    loss = self.loss_function(output.view(-1, ntokens), targets)
                    loss.backward()

                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                    optimizer.step()

                    total_loss += loss.data

                    if batch % self.log_interval == 0 and batch > 0:
                        cur_loss = total_loss.item() / self.log_interval
                        elapsed = time.time() - start_time
                        log.info('| split {:3d} /{:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                              'loss {:5.2f} | ppl {:8.2f}'.format(
                            split, number_of_splits, batch, len(train_data) // sequence_length,
                                                            elapsed * 1000 / self.log_interval, cur_loss,
                            math.exp(cur_loss)))
                        total_loss = 0
                        start_time = time.time()

                log.info('training done! \t({:%H:%M:%S})'.format(datetime.datetime.now()))

                ###############################################################################
                # TEST
                ###############################################################################
                self.model.eval()
                val_loss = self.evaluate(val_data, mini_batch_size, sequence_length)
                scheduler.step(val_loss)

                log.info('best loss so far {:5.2f}'.format(best_val_loss))

                log.info(self.model.generate_text())

                if checkpoint:
                    self.model.save_checkpoint(base_path / 'checkpoint.pt', optimizer, epoch, split, best_val_loss)

                # Save the model if the validation loss is the best we've seen so far.
                if val_loss < best_val_loss:
                    self.model.best_score = best_val_loss
                    self.model.save(savefile)
                    best_val_loss = val_loss

                ###############################################################################
                # print info
                ###############################################################################
                log.info('-' * 89)

                local_split_number = split % number_of_splits
                if local_split_number == 0: local_split_number = number_of_splits

                summary = '| end of split {:3d} /{:3d} | epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' \
                          'valid ppl {:8.2f} | learning rate {:3.4f}'.format(local_split_number,
                                                                             number_of_splits,
                                                                             epoch,
                                                                             (time.time() - epoch_start_time),
                                                                             val_loss,
                                                                             math.exp(val_loss),
                                                                             learning_rate)

                with open(loss_txt, "a") as myfile:
                    myfile.write('%s\n' % summary)

                log.info(summary)
                log.info('-' * 89)

        except KeyboardInterrupt:
            log.info('-' * 89)
            log.info('Exiting from training early')

        ###############################################################################
        # final testing
        ###############################################################################
        test_data = self._batchify(self.corpus.test, mini_batch_size)
        test_loss = self.evaluate(test_data, mini_batch_size, sequence_length)

        summary = 'TEST: valid loss {:5.2f} | valid ppl {:8.2f}'.format(test_loss, math.exp(test_loss))
        with open(loss_txt, "a") as myfile:
            myfile.write('%s\n' % summary)

        log.info(summary)
        log.info('-' * 89)

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
        data = Variable(source[i:i + seq_len])
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        return data, target

    @staticmethod
    def _repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        return tuple(Variable(v) for v in h)

    @staticmethod
    def load_from_checkpoint(checkpoint_file: Path, corpus: TextCorpus, optimizer: Optimizer = SGD):
        checkpoint = LanguageModel.load_checkpoint(checkpoint_file)
        return LanguageModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'],
                                    split=checkpoint['split'], loss=checkpoint['loss'],
                                    optimizer_state=checkpoint['optimizer_state_dict'])
