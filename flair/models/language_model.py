from pathlib import Path

import torch.nn as nn
import torch
import math
from typing import Union, Tuple
from typing import List

from torch.optim import Optimizer

import flair
from flair.data import Dictionary


class LanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        dictionary: Dictionary,
        is_forward_lm: bool,
        hidden_size: int,
        nlayers: int,
        embedding_size: int = 100,
        nout=None,
        document_delimiter: str = '\n',
        dropout=0.1,
    ):

        super(LanguageModel, self).__init__()

        self.dictionary = dictionary
        self.document_delimiter = document_delimiter
        self.is_forward_lm: bool = is_forward_lm

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(len(dictionary), embedding_size)

        if nlayers == 1:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)

        self.hidden = None

        self.nout = nout
        if nout is not None:
            self.proj = nn.Linear(hidden_size, nout)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(nout, len(dictionary))
        else:
            self.proj = None
            self.decoder = nn.Linear(hidden_size, len(dictionary))

        self.init_weights()

        # auto-spawn on GPU if available
        self.to(flair.device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.detach().uniform_(-initrange, initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_(-initrange, initrange)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, hidden, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(emb, hidden)

        if self.proj is not None:
            output = self.proj(output)

        output = self.drop(output)

        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )

        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            output,
            hidden,
        )

    def init_hidden(self, bsz):
        weight = next(self.parameters()).detach()
        return (
            weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
            weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
        )

    def get_representation(
        self,
        strings: List[str],
        start_marker: str,
        end_marker: str,
        chars_per_chunk: int = 512,
    ):

        len_longest_str: int = len(max(strings, key=len))

        # pad strings with whitespaces to longest sentence
        padded_strings: List[str] = []

        for string in strings:
            if not self.is_forward_lm:
                string = string[::-1]

            padded = f"{start_marker}{string}{end_marker}"
            padded_strings.append(padded)

        # cut up the input into chunks of max charlength = chunk_size
        chunks = []
        splice_begin = 0
        longest_padded_str: int = len_longest_str + len(start_marker) + len(end_marker)
        for splice_end in range(chars_per_chunk, longest_padded_str, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in padded_strings])
            splice_begin = splice_end

        chunks.append(
            [text[splice_begin:longest_padded_str] for text in padded_strings]
        )
        hidden = self.init_hidden(len(chunks[0]))

        padding_char_index = self.dictionary.get_idx_for_item(" ")

        batches: List[torch.Tensor] = []
        # push each chunk through the RNN language model
        for chunk in chunks:
            len_longest_chunk: int = len(max(chunk, key=len))
            sequences_as_char_indices: List[List[int]] = []
            for string in chunk:
                char_indices = self.dictionary.get_idx_for_items(list(string))
                char_indices += [padding_char_index] * (len_longest_chunk - len(string))

                sequences_as_char_indices.append(char_indices)
            t = torch.tensor(sequences_as_char_indices, dtype=torch.long).to(
                device=flair.device, non_blocking=True
            )
            batches.append(t)

        output_parts = []
        for batch in batches:
            batch = batch.transpose(0, 1)
            _, rnn_output, hidden = self.forward(batch, hidden)
            output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output

    def get_output(self, text: str):
        char_indices = [self.dictionary.get_idx_for_item(char) for char in text]
        input_vector = torch.LongTensor([char_indices]).transpose(0, 1)

        hidden = self.init_hidden(1)
        prediction, rnn_output, hidden = self.forward(input_vector, hidden)

        return self.repackage_hidden(hidden)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == torch.Tensor:
            return h.clone().detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    @staticmethod
    def initialize(matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3.0 / (in_ + out_))
        matrix.detach().uniform_(-stdv, stdv)

    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):

        state = torch.load(str(model_file), map_location=flair.device)

        document_delimiter = state["document_delimiter"] if "document_delimiter" in state else '\n'

        model = LanguageModel(
            dictionary=state["dictionary"],
            is_forward_lm=state["is_forward_lm"],
            hidden_size=state["hidden_size"],
            nlayers=state["nlayers"],
            embedding_size=state["embedding_size"],
            nout=state["nout"],
            document_delimiter=document_delimiter,
            dropout=state["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model

    @classmethod
    def load_checkpoint(cls, model_file: Path):
        state = torch.load(str(model_file), map_location=flair.device)

        epoch = state["epoch"] if "epoch" in state else None
        split = state["split"] if "split" in state else None
        loss = state["loss"] if "loss" in state else None
        document_delimiter = state["document_delimiter"] if "document_delimiter" in state else '\n'

        optimizer_state_dict = (
            state["optimizer_state_dict"] if "optimizer_state_dict" in state else None
        )

        model = LanguageModel(
            dictionary=state["dictionary"],
            is_forward_lm=state["is_forward_lm"],
            hidden_size=state["hidden_size"],
            nlayers=state["nlayers"],
            embedding_size=state["embedding_size"],
            nout=state["nout"],
            document_delimiter=document_delimiter,
            dropout=state["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return {
            "model": model,
            "epoch": epoch,
            "split": split,
            "loss": loss,
            "optimizer_state_dict": optimizer_state_dict,
        }

    def save_checkpoint(
        self, file: Path, optimizer: Optimizer, epoch: int, split: int, loss: float
    ):
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "split": split,
            "loss": loss,
        }

        torch.save(model_state, str(file), pickle_protocol=4)

    def save(self, file: Path):
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
        }

        torch.save(model_state, str(file), pickle_protocol=4)

    def generate_text(
        self,
        prefix: str = "\n",
        number_of_characters: int = 1000,
        temperature: float = 1.0,
        break_on_suffix=None,
    ) -> Tuple[str, float]:

        if prefix == "":
            prefix = "\n"

        with torch.no_grad():
            characters = []

            idx2item = self.dictionary.idx2item

            # initial hidden state
            hidden = self.init_hidden(1)

            if len(prefix) > 1:

                char_tensors = []
                for character in prefix[:-1]:
                    char_tensors.append(
                        torch.tensor(self.dictionary.get_idx_for_item(character))
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

                input = torch.cat(char_tensors).to(flair.device)

                prediction, _, hidden = self.forward(input, hidden)

            input = (
                torch.tensor(self.dictionary.get_idx_for_item(prefix[-1]))
                .unsqueeze(0)
                .unsqueeze(0)
            )

            log_prob = 0.0

            for i in range(number_of_characters):

                input = input.to(flair.device)

                # get predicted weights
                prediction, _, hidden = self.forward(input, hidden)
                prediction = prediction.squeeze().detach()
                decoder_output = prediction

                # divide by temperature
                prediction = prediction.div(temperature)

                # to prevent overflow problem with small temperature values, substract largest value from all
                # this makes a vector in which the largest value is 0
                max = torch.max(prediction)
                prediction -= max

                # compute word weights with exponential function
                word_weights = prediction.exp().cpu()

                # try sampling multinomial distribution for next character
                try:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                except:
                    word_idx = torch.tensor(0)

                # print(word_idx)
                prob = decoder_output[word_idx]
                log_prob += prob

                input = word_idx.detach().unsqueeze(0).unsqueeze(0)
                word = idx2item[word_idx].decode("UTF-8")
                characters.append(word)

                if break_on_suffix is not None:
                    if "".join(characters).endswith(break_on_suffix):
                        break

            text = prefix + "".join(characters)

            log_prob = log_prob.item()
            log_prob /= len(characters)

            if not self.is_forward_lm:
                text = text[::-1]

            return text, log_prob

    def calculate_perplexity(self, text: str) -> float:

        if not self.is_forward_lm:
            text = text[::-1]

        # input ids
        input = torch.tensor(
            [self.dictionary.get_idx_for_item(char) for char in text[:-1]]
        ).unsqueeze(1)
        input = input.to(flair.device)

        # push list of character IDs through model
        hidden = self.init_hidden(1)
        prediction, _, hidden = self.forward(input, hidden)

        # the target is always the next character
        targets = torch.tensor(
            [self.dictionary.get_idx_for_item(char) for char in text[1:]]
        )
        targets = targets.to(flair.device)

        # use cross entropy loss to compare output of forward pass with targets
        cross_entroy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entroy_loss(
            prediction.view(-1, len(self.dictionary)), targets
        ).item()

        # exponentiate cross-entropy loss to calculate perplexity
        perplexity = math.exp(loss)

        return perplexity

    def _apply(self, fn):
        major, minor, build, *_ = (int(info)
                                for info in torch.__version__.replace("+",".").split('.') if info.isdigit())

        # fixed RNN change format for torch 1.4.0
        if major >= 1 and minor >= 4:
            for child_module in self.children():
                if isinstance(child_module, torch.nn.RNNBase):
                    _flat_weights_names = []
                    num_direction = None

                    if child_module.__dict__["bidirectional"]:
                        num_direction = 2
                    else:
                        num_direction = 1
                    for layer in range(child_module.__dict__["num_layers"]):
                        for direction in range(num_direction):
                            suffix = "_reverse" if direction == 1 else ""
                            param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                            if child_module.__dict__["bias"]:
                                param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                            param_names = [
                                x.format(layer, suffix) for x in param_names
                            ]
                            _flat_weights_names.extend(param_names)

                    setattr(child_module, "_flat_weights_names",
                            _flat_weights_names)

                child_module._apply(fn)

        else:
            super()._apply(fn)
