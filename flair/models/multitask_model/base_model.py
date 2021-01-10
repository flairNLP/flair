import logging
from typing import Union, List, Optional

import torch.nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import flair.nn
from flair.embeddings import TokenEmbeddings
from flair.data import Sentence, Dictionary

from .crf import CRF, ViterbiLoss
from .utils import get_tag_list

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

class BaseModel(torch.nn.Module):

    def __init__(self,
                 embeddings: TokenEmbeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 reproject_embeddings: bool = True,
                 use_rnn: bool = True,
                 rnn_type: str = "LSTM",
                 hidden_size: int = 256,
                 rnn_layers: int = 1,
                 bidirectional: bool = True,
                 use_crf: bool = True,
                 use_lm: bool = True,
                 dropout: float = 0.0,
                 word_dropout: float = 0.05,
                 locked_dropout: float = 0.5):

        super(BaseModel, self).__init__()

        self.embeddings = embeddings
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)
        embedding_dim: int = self.embeddings.embedding_length
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        if reproject_embeddings or use_rnn:
            rnn_input_dim: int = embedding_dim

        self.use_crf = use_crf
        self.use_lm = use_lm

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        self.reproject_embeddings = reproject_embeddings
        if reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim: int = self.reproject_embeddings
            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        if rnn_type in ["LSTM", "GRU", "RNN"]:
            self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim)
        else:
            self.linear = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        if use_crf:
            num_directions = 2 if bidirectional else 1
            self.crf = CRF(hidden_size * num_directions, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary).to(flair.device)
        else:
            self.linear2tag = torch.nn.Linear(rnn_input_dim, len(tag_dictionary))
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss().to(flair.device)

    @staticmethod
    def RNN(rnn_type: str, rnn_layers: int, hidden_size: int, bidirectional: bool, rnn_input_dim: int):
        RNN = getattr(torch.nn, rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=rnn_layers,
                dropout=0.0 if rnn_layers == 1 else 0.5,
                bidirectional=bidirectional,
                batch_first=True,
            )

        return RNN

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]):
        features = self.forward(sentences)
        return self.loss(features)

    def forward(self, sentences):

        self.embeddings.embed(sentences)

        tensor_list = list()
        for sentence in sentences:
            tensor_list.append(sentence.get_sequence_tensor())
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        # Feedforward part
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
            packed = pack_padded_sequence(sentence_tensor, lengths, enforce_sorted=False, batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            pass

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features


    def loss(self, features, sentences):

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = get_tag_list(sentences, self.tag_type)

        if self.use_crf:
            loss = self.viterbi_loss(features, tag_list, lengths)
        else:
            loss = self.cross_entropy_loss(features, tag_list)

        return loss