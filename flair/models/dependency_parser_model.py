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


class DependencyParser(flair.nn.Model):
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
    ):
        """
        Initializes a DependecyParser
        The model is based on biaffine dependency parser :cite: "Dozat T. & Manning C. Deep biaffine attention for neural dependency parsing."
        
        :param token_embeddings: word embeddings used in model
        :param relations_dictionary: dictionary of relations tags
        :param lstm_hidden_size: size of LSTM hidden state
        :param mlp_arc_units: size of MLP for arc 
        :param mlp_rel_units: size of MLP for dependency relations
        :param lstm_layers: number of LSTM layers
        :param mlp_dropout: The dropout probability of MLP layers
        :param lstm_dropout: dropout probability in LSTM 
        """
        
        super(DependencyParser, self).__init__()
        self.token_embeddings = token_embeddings
        self.relations_dictionary: Dictionary = relations_dictionary
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

    def forward(self, sentences: List[Sentence]):
        self.token_embeddings.embed(sentences)
        batch_size = len(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        seq_len: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.token_embeddings.embedding_length * seq_len,
            dtype=torch.float,
            device=flair.device,
        )

        # embed sentences
        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = seq_len - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.token_embeddings.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                batch_size,
                seq_len,
                self.token_embeddings.embedding_length,
            ]
        )

        x = pack_padded_sequence(sentence_tensor, lengths, True, False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)

        # apply MLPs for arc and relations to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get scores from the biaffine attentions
        # [batch_size, seq_len, seq_len]
        score_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        score_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return score_arc, score_rel
    

    def forward_loss(self, data_points: List[Sentence]) -> torch.tensor:
        
        score_arc, score_rel = self.forward(data_points)
        loss_arc, loss_rel = self._calculate_loss(
            score_arc, score_rel, data_points)
        main_loss = loss_arc + loss_rel

        return main_loss


    def _calculate_loss(self, score_arc: torch.tensor,
                        score_relation: torch.tensor,
                        data_points: List[Sentence]) -> Tuple[float, float]:

        arc_loss = 0.0
        rel_loss = 0.0

        for sen_id, sen in enumerate(data_points):

            arc_labels = [token.head_id - 1 if token.head_id !=
                          0 else token.idx - 1 for token in sen.tokens]
            arc_labels = torch.tensor(
                arc_labels, dtype=torch.int64, device=flair.device)
            arc_loss += self.loss_function(score_arc[sen_id], arc_labels)

            rel_labels = [self.relations_dictionary.get_idx_for_item(token.get_tag('dependency').value)
                          for token in sen.tokens]
            rel_labels = torch.tensor(
                rel_labels, dtype=torch.int64, device=flair.device)
            score_relation = score_relation[sen_id][torch.arange(
                len(arc_labels)), arc_labels]
            rel_loss += self.loss_function(score_relation, rel_labels)

        return arc_loss, rel_loss
