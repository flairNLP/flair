import logging
from typing import Union, List

import torch.nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import flair.nn
from flair.data import Sentence, Dictionary
from flair.embeddings import TokenEmbeddings
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG

from .crf import CRF
from .utils import get_tags_tensor

log = logging.getLogger("flair")

class BaseModel(flair.nn.Model):
    """
    Basic multitask model.
    """

    def __init__(
         self,
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
         dropout: float = 0.0,
         word_dropout: float = 0.05,
         locked_dropout: float = 0.5
    ):
        """
        Initializes a base multitask model instance
        :param embeddings: embeddings which are used
        :param tag_dictionary: Dictionary of tags of task
        :param tag_type: Type of tag which is going to be predicted
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        :param use_rnn: if True, adds a RNN layer to the model. If False, simple linear layer.
        :param rnn_type: specifies the RNN type to use. Use "RNN", "GRU" or "LSTM". Default is "LSTM".
        :param hidden_size: hidden size of the rnn layer.
        :param rnn_layers: number of layers to use for RNN.
        :param bidirectional: If True, RNN layer is bidirectional. If False, single direction.
        :param use_crf: If True, use Conditonal Random Field. If False, use Dense Softmax layer for prediction.
        :param use_lm: If True, use additional language model during training for multitask purpose.
        :param dropout: Includes standard dropout, if provided attribute value is > 0.0
        :param word_dropout: Includes word_dropout, if provided attribute value is > 0.0
        :param locked_dropout: Includes locked_dropout, if provided attribute value is > 0.0
        """

        super(BaseModel, self).__init__()

        # Embeddings and task specific attributes
        self.embeddings = embeddings
        self.tag_dictionary = tag_dictionary
        self.tag_type = tag_type
        self.tagset_size: int = len(tag_dictionary)
        embedding_dim: int = self.embeddings.embedding_length
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # RNN specific attributes
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        if reproject_embeddings or use_rnn:
            rnn_input_dim: int = embedding_dim

        # CRF and LM specific attributes
        self.use_crf = use_crf

        # Dropout specific attributes
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout= True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # Model layers
        # Main task
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        self.reproject_embeddings = reproject_embeddings
        if reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim: int = self.reproject_embeddings
            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        if use_rnn:
            self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim)
        else:
            self.linear = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        num_directions = 2 if bidirectional else 1
        if use_crf:
            self.crf = CRF(hidden_size * num_directions, tag_dictionary, self.tagset_size)
        else:
            self.linear2tag = torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))
            self.loss_criterion = torch.nn.CrossEntropyLoss()

        self.to(flair.device)

    @staticmethod
    def RNN(
        rnn_type: str,
        rnn_layers: int,
        hidden_size: int,
        bidirectional: bool,
        rnn_input_dim: int
    ):
        """
        Static wrapper function returning an RNN instance from PyTorch
        :param rnn_type: Type of RNN from torch.nn
        :param rnn_layers: number of layers to include
        :param hidden_size: hidden size of RNN cell
        :param bidirectional: If True, RNN cell is bidirectional
        :param rnn_input_dim: Input dimension to RNN cell
        """
        if rnn_type in ["LSTM", "GRU", "RNN"]:
            RNN = getattr(torch.nn, rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=rnn_layers,
                    dropout=0.0 if rnn_layers == 1 else 0.5,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
        else:
            raise Exception(f"Unknown RNN type: {rnn_type}. Please use either LSTM, GRU or RNN.")

        return RNN

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        features = self.forward(sentences)
        return self.loss(features, sentences)

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        """

        # preparation of sentences and feed-forward part
        self.embeddings.embed(sentences)

        lengths = torch.LongTensor([len(sentence.tokens) for sentence in sentences])

        tensor_list = list()
        for sentence in sentences:
            tensor_list.append(sentence.get_sequence_tensor())
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        # feed-forward part
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
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


    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Loss function of multitask base model.
        :param features: Output features / CRF scores from feed-forward function
        :param sentences: list of sentences
        """

        # Preparation for loss function
        lengths = torch.LongTensor([len(sentence.tokens) for sentence in sentences])
        tags_tensor = get_tags_tensor(sentences, self.tag_dictionary, self.tag_type)

        # If CRF is used, features come as CRF scores. Use forward_alg and gold_score function to get loss
        # If no CRF, use cross_entropy_loss.
        if self.use_crf:
            forward_score = self.crf.forward_alg(features, lengths)
            gold_score = self.crf.gold_score(features, tags_tensor, lengths)
            loss = (forward_score - gold_score).mean()
        else:
            loss = self.loss_criterion(features.permute(0,2,1), tags_tensor)

        return loss