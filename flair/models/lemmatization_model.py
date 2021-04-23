import logging
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

import flair.nn
from flair.data import Dictionary, Sentence, Token
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import FlairEmbeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import Metric, Result, store_embeddings

SOS_token:str = '<SOS>'  # Start-of-sentence token
EOS_token:str = '<EOS>'  # End-of-sentence token

class Lemma(flair.nn.Model):
    def __init__(
            self,
            hidden_size: int,
            pre_embeddings: FlairEmbeddings,
            character_dictionary:Dictionary,
            n_layers: int,
            tag_list: list = [],
            tag_dictionary: Dictionary = Dictionary(),
            dropout: float = 0.1,
            rnn_type: str = "GRU",
    ):

        super(Lemma, self).__init__()
        self.hidden_size = hidden_size
        self.pre_embeddings = pre_embeddings
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.character_dictionary = character_dictionary
        self.character_dictionary.add_item(SOS_token)
        self.character_dictionary.add_item(EOS_token)

        self.pre_embedding_weight = self._load_embedding_weight(self.character_dictionary, self.pre_embeddings)
        self.embedding = nn.Embedding(len(self.character_dictionary), self.pre_embeddings.embedding_length)
        self.embedding.weight.data.copy_(self.pre_embedding_weight)
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(dropout)

        if rnn_type is "GRU":
            self.encoder = nn.GRU(self.pre_embeddings.embedding_length,
                                  hidden_size,
                                  n_layers,
                                  dropout=(0 if n_layers == 1 else dropout),
                                  bidirectional=True)
            self.decoder = nn.GRU(self.pre_embeddings.embedding_length,
                                  hidden_size,
                                  n_layers,
                                  dropout=(0 if n_layers == 1 else dropout))
        elif rnn_type is "LSTM":
            self.encoder = nn.LSTM(self.pre_embeddings.embedding_length,
                                   hidden_size,
                                   n_layers,
                                   dropout=(0 if n_layers == 1 else dropout))
            self.decoder = nn.LSTM(self.pre_embeddings.embedding_length,
                                  hidden_size,
                                  n_layers,
                                  dropout=(0 if n_layers == 1 else dropout))
        else:
            print("Only LSTM and GRU are supported for now")
        # for decoder
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, len(self.character_dictionary))

        # If tag_list is not empty, it is considered that the user needs to use tag.
        if len(tag_list):
            self.use_tag:bool = True
            self.tag_list = tag_list
            self.tag_dictionary = tag_dictionary
            self.tag_embedding = nn.Embedding(len(self.tag_dictionary), self.pre_embeddings.embedding_length)
            self.tag_embedding.weight.data.uniform_(-1, 1)
        else:
            self.use_tag:bool = False

        self.max_length:int = 0

        self.to(flair.device)

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        loss = self.forward(data_points)
        return  loss
    def forward(self, sentences: List[Sentence]):

        for sentence in sentences:
            loss = 0
            encoder_input, tag, lemma, mask, input_lenghts, max_lemma_lenght = self._preprocessed_data(sentence)

            encoder_out, encoder_hidden = self._encode(encoder_input, tag, input_lenghts)

            decoder_input = torch.LongTensor([[self.character_dictionary.get_idx_for_item(SOS_token) for _ in range(len(input_lenghts))]])
            decoder_input = decoder_input.to(flair.device)

            decoder_hidden = encoder_hidden[:self.n_layers]

            for t in range(max_lemma_lenght):
                decoder_output, decoder_hidden = self._decode(
                    decoder_input, decoder_hidden, encoder_out
                )

                decoder_input = lemma[t].view(1, -1)
                mask_loss, nTotal = self._maskNLLLoss(decoder_output, lemma[t], mask[t])
                loss += mask_loss
        return loss

    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            wsd_evaluation: bool = False,
            **kwargs
    ) -> (Result, float):

        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        for batch in data_loader:
            for sentence in batch:
                for token in sentence:
                    decode_seq = self.predict(token)
                    decode_word = [self.character_dictionary.get_item_for_index(ch.item())for ch in decode_seq]
                    decode_word[:] = [x for x in decode_word if not (x == EOS_token or x == '<unk>')]
                    output = ''.join(decode_word)
                    print(output)


    def predict(self, input):
        if type(input) is Token:
            input_seq = self._item2seq(self.character_dictionary, input.text)
            input_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))
            if self.use_tag:
                tags = []
                for tag in self.tag_list:
                    tags.append(input.get_tag(tag).value)
            tags_seq = self._item2seq(self.tag_dictionary, tags)
            if len(tags_seq) > 1:
                tags_seq = torch.LongTensor([tags_seq]).transpose(0, 1)
            else:
                tags_seq = torch.LongTensor([tags_seq])
            tags_seq.to(flair.device)
            input_lenghts = torch.tensor([len(input_seq)]).to(device = 'cpu')
            encoder_input = torch.LongTensor([input_seq]).transpose(0, 1)
            encoder_input.to(flair.device)
        if type(input) is Sentence:
            self._preprocessed_data(input)
            encoder_input, tags_seq, lemma, _, input_lenghts, _ = self._preprocessed_data(input)

        encoder_outputs, encoder_hidden = self._encode(encoder_input, tags_seq, input_lenghts)
        decoder_hidden = encoder_hidden[:self.n_layers]
        decoder_input = torch.ones(1, 1, device=flair.device, dtype=torch.long) * self.character_dictionary.get_idx_for_item(SOS_token)
        all_chars = torch.zeros([0], device=flair.device, dtype=torch.long)
        # all_scores = torch.zeros([0], device=flair.device)

        for _ in range(0, self.max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_chars = torch.cat((all_chars, decoder_input), dim=0)
            # all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_chars


    def _encode(self, encoder_input, tag, input_lengths):
        hidden = None
        encoder_input = self.embedding(encoder_input)
        if self.use_tag:
            tag_emb = self.tag_embedding(tag)
            encoder_input = torch.cat((encoder_input, tag_emb), dim=0)
        packed = nn.utils.rnn.pack_padded_sequence(encoder_input, input_lengths)
        outputs, hidden = self.encoder(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        if self.rnn_type is "GRU":
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def _attn(self, hidden, encoder_outputs, method = 'dot'):
        # Implement the dot method first
        if method == 'dot':
            attn_energies = self._dot_score(hidden, encoder_outputs)
            attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def _dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def _decode(self,  input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.decoder(embedded, last_hidden)
        attn_weights = self._attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden

    def _maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()

        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(flair.device)
        return loss, nTotal.item()

    def _load_embedding_weight(self, dictionary, pre_embedding):
        embeddings = torch.zeros((len(dictionary), pre_embedding.embedding_length))
        embeddings.to(flair.device)
        for i in range(0, len(dictionary)):
            sentence = Sentence(dictionary.get_item_for_index(i))
            if len(sentence) is not 0:
                pre_embedding.embed(sentence)
                embeddings[i] = sentence[0].embedding

        return embeddings

    def _item2seq(self, dictionary, input):
        # Convert item into a sequence.
        sequence = []
        for item in input:
            sequence.append(dictionary.get_idx_for_item(item))

        return sequence

    def _preprocessed_data(self, sentence):
        """
        Preprocess the data, convert each token of the input sentence into a sequence of the original word,
        tag sequence, and lemma sequence. At the same time, the length of the sequence is unified by zero padding.
        Return as tensor.
        :return: word_seq_list, tag_seq_list, lemma_seq_list, mask, word_seq_lenght
        word_seq_list: tensor([[word1_seq],
                               [word2_seq],
                               [word3_seq],
                               ...])
        tag_seq_list and lemma_seq_list are the same as word_seq_list.
        mask: BoolTensor. The zero-padded part of the sequence should be ignored when calculating the loss
        word_seq_lenght: Record the original length of the input sequence.
        """
        tokens = sentence.tokens[:]
        tokens.sort(key=lambda x: len(x.text), reverse=True)

        word_seq_list = []
        tag_seq_list = []
        lemma_seq_list = []
        max_word_lenght = 0
        max_lemma_lenght = 0
        sentence.tokens
        word_seq_lenght = []

        for token in tokens:
            # character sequence of token
            word_seq = self._item2seq(self.character_dictionary, token.text)

            # add end symbol
            word_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))
            max_word_lenght = max(max_word_lenght, len(word_seq))
            word_seq_lenght.append(len(word_seq))
            word_seq_list.append(word_seq)

            tags_seq = []
            if self.use_tag:
                for tag in self.tag_list:
                    tags_seq.append(token.get_tag(tag).value)
                tags_seq = self._item2seq(self.tag_dictionary, tags_seq)
            tag_seq_list.append(tags_seq)

            lemma_seq = self._item2seq(self.character_dictionary, token.get_tag('lemma').value)
            lemma_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))

            max_lemma_lenght = max(max_lemma_lenght, len(lemma_seq))
            lemma_seq_list.append(lemma_seq)

        # Zero padding
        word_seq_list = self._seq_zero_padding(word_seq_list, max_word_lenght)
        # Transpose input data
        word_seq_list = torch.LongTensor(word_seq_list).transpose(0, 1)

        tag_seq_list = torch.LongTensor(tag_seq_list).transpose(0, 1)

        lemma_seq_list = self._seq_zero_padding(lemma_seq_list, max_lemma_lenght)
        # lemma_seq_list = list(map(list, zip(*lemma_seq_list)))
        lemma_seq_list = torch.LongTensor(lemma_seq_list).transpose(0, 1)

        mask = torch.BoolTensor(self._binary_matrix(lemma_seq_list))

        self.max_length = max(self.max_length, max_word_lenght, max_lemma_lenght)

        return word_seq_list, tag_seq_list, lemma_seq_list, mask, word_seq_lenght, max_lemma_lenght

    def _seq_zero_padding(self, seqs:list, target_length:int, fillvalue=0):
        for seq in seqs:
            seq.extend([fillvalue] * (target_length - len(seq)))
        return seqs

    def _binary_matrix(self, seqs:list):
        matrix = []
        for i, seq in enumerate(seqs):
            matrix.append([])
            for value in seq:
                if value == 0:
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)
        return matrix
    # @staticmethod
    # def _init_model_with_state_dict(state):
    #     model = Lemma(
    #         hidden_size=state["hidden_size"],
    #         # embeddings=state["embeddings"],
    #         # tag_dictionary=state["tag_dictionary"],
    #         # tag_type=state["tag_type"],
    #         # use_crf=state["use_crf"],
    #         # rnn_layers=state["rnn_layers"],
    #         # dropout=use_dropout,
    #         # word_dropout=use_word_dropout,
    #         # locked_dropout=use_locked_dropout,
    #         # train_initial_hidden_state=train_initial_hidden_state,
    #         # rnn_type=rnn_type,
    #         # beta=beta,
    #         # loss_weights=weights,
    #         # reproject_embeddings=reproject_embeddings,
    #     )
    #     model.load_state_dict(state["state_dict"])
    #     return model