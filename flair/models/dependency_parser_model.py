import enum
from pathlib import Path
from typing import List, Union, Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence
import sklearn
import numpy as np

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, DataPoint
from flair.datasets import DataLoader, SentenceDataset
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result, store_embeddings
from flair.nn.dropout import LockedDropout
from flair.visual.tree_printer import tree_printer

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DependecyParser(flair.nn.Model):
    def __init__(
            self,
            token_embeddings: TokenEmbeddings,
            relations_dictionary: Dictionary,
            lstm_hidden_size: int = 400,
            mlp_arc_units: int = 500,
            mlp_rel_units: int = 100,
            lstm_layers: int = 3,
            mlp_dropout: float = 0.1,
            lstm_dropout: float = 0.2,
            relearn_embeddings: bool = True,
    ):

        super(DependecyParser, self).__init__()
        self.token_embeddings = token_embeddings
        self.relations_dictionary: Dictionary = relations_dictionary
        self.relearn_embeddings = relearn_embeddings
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_arc_units = mlp_arc_units
        self.mlp_rel_units = mlp_rel_units
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout
        self.tag_type = 'dependency'
        self.lstm_input_dim: int = self.token_embeddings.embedding_length
        if self.relations_dictionary:
            self.embedding2nn = torch.nn.Linear(
                self.lstm_input_dim, self.lstm_input_dim)

        self.lstm = BiLSTM(input_size=self.lstm_input_dim,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.lstm_dropout)

        self.mlp_arc_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_arc_units,
                             dropout=self.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_arc_units,
                             dropout=self.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_rel_units,
                             dropout=self.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_rel_units,
                             dropout=self.mlp_dropout)

        self.arc_attn = Biaffine(n_in=self.mlp_arc_units,
                                 bias_x=True,
                                 bias_y=False)

        self.rel_attn = Biaffine(n_in=self.mlp_rel_units,
                                 n_out=len(relations_dictionary),
                                 bias_x=True,
                                 bias_y=True)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.to(flair.device)
