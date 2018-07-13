import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from typing import Dict, List
from .data import Dictionary


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nout, nlayers, dropout=0.5):

        super(RNNModel, self).__init__()

        self.dictionary = Dictionary()
        self.is_forward_lm: bool = True

        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if nlayers == 1:
            self.rnn = nn.LSTM(ninp, nhid, nlayers)
        else:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

        self.hidden = None

        if nout is not None:
            self.proj = nn.Linear(nhid, nout)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(nout, ntoken)
        else:
            self.proj = None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, hidden, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)

        output, hidden = self.rnn(emb, hidden)

        if self.proj is not None:
            output = self.proj(output)

        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def get_representation(self, strings: List[str], detach_from_lm=True):

        sequences_as_char_indices: List[List[int]] = []
        for string in strings:
            char_indices = [self.dictionary.get_idx_for_item(char) for char in string]
            sequences_as_char_indices.append(char_indices)

        batch = Variable(torch.LongTensor(sequences_as_char_indices).transpose(0, 1))

        if torch.cuda.is_available():
            batch = batch.cuda()

        hidden = self.init_hidden(len(strings))
        prediction, rnn_output, hidden = self.forward(batch, hidden)

        if detach_from_lm: rnn_output = self.repackage_hidden(rnn_output)

        return rnn_output

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == torch.Tensor:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initialize(self, matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3. / (in_ + out_))
        matrix.data.uniform_(-stdv, stdv)

    @classmethod
    def load_language_model(cls, model_file):
        state = torch.load(model_file)
        model = RNNModel(state['rnn_type'], state['ntoken'], state['ninp'], state['nhid'], state['nout'],
                         state['nlayers'], state['dropout'])
        model.load_state_dict(state['state_dict'])
        model.is_forward_lm = state['is_forward_lm']
        model.dictionary = state['char_dictionary_forward']
        return model

    def save(self, file):
        model_state = {
            'state_dict': self.state_dict(),
            'is_forward_lm': self.is_forward_lm,
            'char_dictionary_forward': self.dictionary,
            'rnn_type': self.rnn_type,
            'ntoken': len(self.dictionary),
            'ninp': self.ninp,
            'nhid': self.nhid,
            'nout': self.proj,
            'nlayers': self.nlayers,
            'dropout': self.dropout
        }
        torch.save(model_state, file, pickle_protocol=4)
