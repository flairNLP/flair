import logging
from pathlib import Path
from typing import List, Union
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

import flair.nn
from flair.data import Dictionary, Sentence, Token
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import FlairEmbeddings
from flair.training_utils import Metric, Result

log = logging.getLogger("flair")


SOS_token: str = '<SOS>'  # Start-of-sentence token
EOS_token: str = '<EOS>'  # End-of-sentence token


class Lemmatization(flair.nn.Model):

    def __init__(
            self,
            hidden_size: int,
            pre_embeddings: FlairEmbeddings,
            character_dictionary: Dictionary,
            n_layers: int,
            tag_list: list = None,
            tag_dictionary: Dictionary = None,
            dropout: float = 0.1,
            teacher_forcing_ratio: float = 0.5,
            longest_word_length: int = 50

    ):
        """
        :param hidden_size: number of hidden states in RNN
        :param pre_embeddings: pre-trained embeddings
        :param character_dictionary: dictionary containing all the characters in the corpus
        :param n_layers: number of RNN layers
        :param tag_list: list of tags planned to be used
        :param tag_dictionary: dictionary of tags you want to predict
        :param dropout: dictionary contains all the tag values in the tag list
        :param teacher_forcing_ratio: the probability of using teacher_forcing. If it is greater than or equal to 1,
        always use teacher forcing. If less than or equal to 0, never use.
        :param longest_word_length: record the maximum length of words and lemma to use in prediction.
        Based on experience, the default value is set at 50.
        """

        super(Lemmatization, self).__init__()
        self.hidden_size = hidden_size
        self.pre_embeddings = pre_embeddings.to(flair.device)
        self.n_layers = n_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
        self.character_dictionary = character_dictionary
        self.character_dictionary.add_item(SOS_token)
        self.character_dictionary.add_item(EOS_token)

        self.pre_embedding_weight = self._load_embedding_weight(self.character_dictionary, self.pre_embeddings)
        self.embedding = nn.Embedding(len(self.character_dictionary), self.pre_embeddings.embedding_length)
        self.embedding.weight.data.copy_(self.pre_embedding_weight)
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(dropout)

        # TODO: LSTM model implementation
        self.encoder = nn.GRU(self.pre_embeddings.embedding_length,
                              hidden_size,
                              n_layers,
                              dropout=(0 if n_layers == 1 else dropout),
                              bidirectional=True)
        self.decoder = nn.GRU(self.pre_embeddings.embedding_length,
                              hidden_size,
                              n_layers,
                              dropout=(0 if n_layers == 1 else dropout))

        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, len(self.character_dictionary))

        # If tag_list is not empty, it is considered that the user needs to use tag.
        if tag_list != None and tag_dictionary != None:
            if len(tag_list) > 0:
                self.use_tag: bool = True
                self.tag_list = tag_list
                self.tag_dictionary = tag_dictionary
                self.tag_embeddings = nn.Embedding(len(self.tag_dictionary), self.pre_embeddings.embedding_length)
                self.tag_embeddings.weight.data.uniform_(-1, 1)
            else:
                self.use_tag: bool = False
                log.info("tag_list is empty")
        elif tag_list == None and tag_dictionary == None:
            self.use_tag: bool = False
        else:
            self.use_tag: bool = False
            log.info("If you want to use tag when training the model, do not forget either of the tag_list and tag_dictionary parameters.")

        # Record the maximum length of words and lemma to use in prediction
        self.longest_word_length: int = longest_word_length

        self.to(flair.device)


    def _load_embedding_weight(self, dictionary: Dictionary, pre_embedding: FlairEmbeddings):
        """
        According to character_dictionary, load weights from pre-trained Embedding.
        Return an embedding that has been loaded with weights and fits our model.
        """
        embeddings = torch.zeros((len(dictionary), pre_embedding.embedding_length)).to(flair.device)
        for i in range(0, len(dictionary)):
            sentence = Sentence(dictionary.get_item_for_index(i))
            if len(sentence) is not 0:
                pre_embedding.embed(sentence)
                embeddings[i] = sentence[0].embedding

        return embeddings

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        loss = self.forward(data_points)
        return loss

    def forward(self, sentences: List[Sentence]):

        for sentence in sentences:
            loss = 0
            encoder_input, tag, lemma, mask, input_lenghts, max_lemma_lenght = self._preprocessed_data(sentence)

            encoder_out, encoder_hidden = self._encode(encoder_input, tag, input_lenghts)

            decoder_hidden = encoder_hidden[:self.n_layers]
            sos_idx = self.character_dictionary.get_idx_for_item(SOS_token)
            decoder_input = torch.LongTensor([[sos_idx for _ in range(len(input_lenghts))]]).to(flair.device)

            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if use_teacher_forcing:
                for t in range(max_lemma_lenght):
                    decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_out)
                    decoder_input = lemma[t].view(1, -1)
                    mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
                    loss += mask_loss
            else:
                for t in range(max_lemma_lenght):
                    decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_out)
                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(len(input_lenghts))]]).to(flair.device)
                    mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
                    loss += mask_loss

        return loss

    def _encode(self, encoder_input, tag, input_lengths):
        hidden = None
        encoder_input = self.embedding(encoder_input)
        if self.use_tag:
            tag_emb = self.tag_embeddings(tag)
            encoder_input = torch.cat((encoder_input, tag_emb), dim=0)
        packed = nn.utils.rnn.pack_padded_sequence(encoder_input, input_lengths)
        outputs, hidden = self.encoder(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def _attn(self, hidden, encoder_outputs, method='dot'):
        # TODO: Add another attention score
        if method == 'dot':
            attn_energies = self._dot_score(hidden, encoder_outputs)
            attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def _dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def _decode(self, input_step, last_hidden, encoder_outputs):
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

    def _calculate_loss(self, input, target, mask):

        cross_entropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean().to(flair.device)

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

        y_true = []
        y_pred = []

        metric = Metric("Evaluation")

        eval_loss = 0
        n = 0
        for batch in data_loader:
            for sentence in batch:
                n += 1
                loss = self._eval_loss(sentence)
                eval_loss += loss
                outputs = self._eval_predict(sentence)
                tokens = sentence.tokens[:]
                tokens.sort(key=lambda x: len(x.text), reverse=True)
                for i in range(0, len(tokens)):
                    y_true.append(tokens[i].get_tag('lemma').value)
                    y_pred.append(outputs[i])

        for i in range(0, len(y_pred)):
            if y_true[i] == y_pred[i]:
                metric.add_tp(y_true[i])
            elif '<unk>' in y_pred[i]:
                metric.add_fn(y_true[i])
            else:
                metric.add_fp(y_true[i])

        detailed_result = (
            "\nResults:"
            f"\n- Accuracy {metric.accuracy()}"
            '\n\nBy class:\n'
        )

        result = Result(
            main_score=metric.accuracy(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        eval_loss /= n

        return result, eval_loss

    def _eval_loss(self, sentence: Sentence):

        loss = 0

        encoder_input, tags_seq, lemma, mask, input_lenghts, max_lemma_lenght = self._preprocessed_data(sentence)

        encoder_outputs, encoder_hidden = self._encode(encoder_input, tags_seq, input_lenghts)
        decoder_hidden = encoder_hidden[:self.n_layers]
        sos_idx = self.character_dictionary.get_idx_for_item(SOS_token)
        decoder_input = torch.LongTensor([[sos_idx for _ in range(len(input_lenghts))]]).to(flair.device)

        for t in range(max_lemma_lenght):
            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_outputs)

            # Take the predicted value with the highest probability as the input for the next moment
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(len(input_lenghts))]]).to(flair.device)

            mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
            loss += mask_loss

        return loss

    def _eval_predict(self, sentence: Sentence):
        """
        In order to improve efficiency during evaluation, predictions are made in the form of batches.
        But it will disrupt the order of tokens in Sentence.
        :param sentence:
        :return: pre_lemmas: The predicted value of the token's lemma.
        """
        encoder_input, tags_seq, _, _, input_lenghts, _ = self._preprocessed_data(sentence)

        all_seqs = self._calculation_output(encoder_input, tags_seq, input_lenghts)

        pre_lemmas = self.seq2chars(all_seqs)

        return  pre_lemmas

    def predict(self, input):
        """
        Use the model to predict lemma.
        :param input: Token or Sentence type. If you want to predict a string, please convert it to the above type.
        :return: Predicted lemma：The result of token prediction is returned as a string, and sentence is returned as a list.
        """

        if type(input) is Token:
            pre_lemma = self._token_predict(input)

            return pre_lemma

        elif type(input) is Sentence:
            pre_lemmas = []
            for token in input:
                pre_lemma = self._token_predict(token)
                pre_lemmas.append(pre_lemma)

            return pre_lemmas
        else:
            log.info("The currently acceptable input types are Token and Sentence.")
            log.info("Adding tags that are compatible with the model can increase the accuracy of prediction.")

            return ''

    def _calculation_output(self, encoder_input, tags_seq, input_lenghts):
        # Calculate the output of the model based on the input, use it in the prediction.

        encoder_outputs, encoder_hidden = self._encode(encoder_input, tags_seq, input_lenghts)
        decoder_hidden = encoder_hidden[:self.n_layers]
        sos_idx = self.character_dictionary.get_idx_for_item(SOS_token)
        decoder_input = torch.LongTensor([[sos_idx for _ in range(len(input_lenghts))]]).to(flair.device)

        all_seqs = torch.zeros((self.longest_word_length, len(input_lenghts)), device=flair.device, dtype=torch.long)

        for i in range(0, self.longest_word_length):
            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_seqs[i] = decoder_input
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # Due to the structure of the input and output data of the model, transpose is needed here.
        all_seqs = all_seqs.transpose(0, 1)

        return all_seqs

    def _token_predict(self, token : Token):
        # Prediction method for the input data of a Token.
        token_seq = self._item2seq(self.character_dictionary, token.text)
        token_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))
        encoder_input = torch.LongTensor([token_seq]).transpose(0, 1).to(flair.device)

        if self.use_tag:
            tags = []
            for tag in self.tag_list:
                tags.append(token.get_tag(tag).value)
            tags_seq = self._item2seq(self.tag_dictionary, tags)
            tags_seq = torch.LongTensor([tags_seq]).to(flair.device)
        else:
            tags_seq = None

        input_lenghts = torch.tensor([len(token_seq)]).to(device='cpu')

        all_seqs = self._calculation_output(encoder_input, tags_seq, input_lenghts)

        pre_lemma = self.seq2chars(all_seqs)

        return pre_lemma

    def seq2chars(self, chars_seqs: List[List]):
        if len(chars_seqs) > 1:
            pre_lemmas = []
            for chars_seq in chars_seqs:
                pre_lemma = ''
                chars = [self.character_dictionary.get_item_for_index(ch.item()) for ch in chars_seq]
                for ch in chars:
                    if ch == EOS_token:
                        break
                    else:
                        pre_lemma = pre_lemma + ch

                pre_lemmas.append(pre_lemma)

            return pre_lemmas

        elif len(chars_seqs) == 1 :
            pre_lemma = ''
            char = [self.character_dictionary.get_item_for_index(ch.item()) for ch in chars_seqs[0]]
            pre_lemma = pre_lemma + char[0]

            return pre_lemma
        else:
            log.info("Sequence is empty.")

            return ''

    def _preprocessed_data(self, sentence: Sentence):
        """
        Preprocess the data, convert each token of the input sentence into a sequence of the original word,
        tag sequence, and lemma sequence. At the same time, the length of the sequence is unified by zero padding.
        Return as tensor. Due to nn.utils.rnn.pack_padded_sequence, the sequence is arranged from long to short.
        :return: word_seq_list, tag_seq_list, lemma_seq_list, mask, word_seq_lenght
        word_seq_list: tensor([[word1_seq[0],word2_seq[0], word3_seq[0], ...],
                               [word1_seq[1],word2_seq[1], word3_seq[1], ...],
                               [word1_seq[2],word2_seq[2], word3_seq[2], ...],
                               ...])
        tag_seq_list and lemma_seq_list are the same as word_seq_list.
        If tag is not used, an empty array is returned as tag_seq_list.
        mask: BoolTensor. The zero-padded part of the sequence should be ignored when calculating the loss
        word_seq_lenght: Record the original length of the input sequence.
        max_lemma_lenght: In this set of data, the maximum length of lemma.(In order to reduce the number of
                        unnecessary iterations during training)
        """
        # Process the input data and sort the tokens in the sentence from longest to shortest according to their length.
        tokens = sentence.tokens[:]
        tokens.sort(key=lambda x: len(x.text), reverse=True)

        word_seq_list = []
        tag_seq_list = []
        lemma_seq_list = []
        max_word_lenght = 0
        max_lemma_lenght = 0
        word_seq_lenght = []

        for token in tokens:
            # Convert the string into a sequence according to the dictionary
            word_seq = self._item2seq(self.character_dictionary, token.text)
            # add end symbol
            word_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))
            # Record the length of each sequence
            word_seq_lenght.append(len(word_seq))
            word_seq_list.append(word_seq)
            max_word_lenght = max(max_word_lenght, len(word_seq))

            if self.use_tag:
                tags_seq = []
                for tag in self.tag_list:
                    tags_seq.append(token.get_tag(tag).value)
                tags_seq = self._item2seq(self.tag_dictionary, tags_seq)
                tag_seq_list.append(tags_seq)

            lemma_seq = self._item2seq(self.character_dictionary, token.get_tag('lemma').value)
            lemma_seq.append(self.character_dictionary.get_idx_for_item(EOS_token))

            max_lemma_lenght = max(max_lemma_lenght, len(lemma_seq))
            lemma_seq_list.append(lemma_seq)

        # Zero padding, make all sequences the same length
        word_seq_list = self._seq_zero_padding(word_seq_list, max_word_lenght)
        # Transpose input data.
        word_seq_list = torch.LongTensor(word_seq_list).transpose(0, 1).to(flair.device)

        if self.use_tag:
            tag_seq_list = torch.LongTensor(tag_seq_list).transpose(0, 1).to(flair.device)

        lemma_seq_list = self._seq_zero_padding(lemma_seq_list, max_lemma_lenght)
        lemma_seq_list = torch.LongTensor(lemma_seq_list).transpose(0, 1).to(flair.device)

        # Generate mask, because when calculating loss, the part through zero padding should not be included in the calculation.
        mask = torch.BoolTensor(self._binary_matrix(lemma_seq_list)).to(flair.device)

        # At the same time, update the length of the longest word encountered for the model.
        self.longest_word_length = max(self.longest_word_length, max_word_lenght, max_lemma_lenght)

        return word_seq_list, tag_seq_list, lemma_seq_list, mask, word_seq_lenght, max_lemma_lenght

    def _item2seq(self, dictionary: Dictionary, input):
        # Convert item into a sequence.
        sequence = []
        for item in input:
            sequence.append(dictionary.get_idx_for_item(item))

        return sequence

    def _seq_zero_padding(self, seqs: list, target_length: int, fill_value=0):
        for seq in seqs:
            seq.extend([fill_value] * (target_length - len(seq)))
        return seqs

    def _binary_matrix(self, seqs: list):
        matrix = []
        for i, seq in enumerate(seqs):
            matrix.append([])
            for value in seq:
                if value == 0:
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)
        return matrix

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "pre_embeddings": self.pre_embeddings,
            "character_dictionary": self.character_dictionary,
            "n_layers": self.n_layers,
            "tag_list": self.tag_list if "tag_list" in self.__dict__.keys() else None,
            "tag_dictionary": self.tag_dictionary if "tag_dictionary" in self.__dict__.keys() else None,
            "dropout": self.dropout,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
            "longest_word_length": self.longest_word_length,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = Lemmatization(
            hidden_size=state["hidden_size"],
            pre_embeddings=state["pre_embeddings"],
            character_dictionary=state["character_dictionary"],
            n_layers=state["n_layers"],
            tag_list=state["tag_list"],
            tag_dictionary=state["tag_dictionary"],
            dropout=state["dropout"],
            teacher_forcing_ratio=state["teacher_forcing_ratio"],
            longest_word_length=state["longest_word_length"],
        )
        model.load_state_dict(state["state_dict"])
        return model
