import logging
from typing import Union, List, Optional

import torch.nn
from torch.nn.utils.rnn import pad_sequence

import flair.nn
from flair.embeddings import TokenEmbeddings
from flair.data import Sentence, Dictionary

log = logging.getLogger("flair")

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
                 dropout: float = 0.0,
                 word_dropout: float = 0.05,
                 locked_dropout: float = 0.5,
                 use_lm: bool = True):
        super(BaseModel, self).__init__()
        self.embeddings = embeddings
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)
        embedding_dim: int = self.embeddings.embedding_length

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        if reproject_embeddings or use_rnn:
            rnn_input_dim: int = embedding_dim

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
            num_directions = 2 if bidirectional else 1
            self.rnn = getattr(torch.nn, rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=rnn_layers,
                dropout=0.0 if rnn_layers == 1 else 0.5,
                bidirectional=True,
                batch_first=True,
            )
            self.linear = torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))

        if use_lm:
            self.use_lm = use_lm
            self.lm_forward = torch.nn.Linear(self.char_rnn_dim, self.lm_vocab_size)
            self.backward = torch.nn.Linear(self.char_rnn_dim, self.lm_vocab_size)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]):
        features = self.forward(sentences)
        return self.loss(features)

    def forward(self, sentences):

        # Prepare batch
        self.embeddings.embed(sentences)

        tensor_list = list()
        for sentence in sentences:
            tensor_list.append(sentence.get_sequence_tensor())  # get tensor of shape (seq_len, embedding_dim)
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

        if self.use_lm:
            lm_f_scores = self.forw_lm_out()
            lm_b_scores = self.back_lm_out()



    def loss(self, feats):
        pass