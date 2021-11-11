import logging
from collections import Counter
from math import inf
from pathlib import Path
from typing import List, Union, Tuple, Optional

import torch
from sklearn.metrics import classification_report, accuracy_score
from torch import nn
from torch.utils.data.dataset import Dataset

import flair.embeddings
import flair.nn
from flair.data import Sentence, Dictionary, Corpus
from flair.datasets import DataLoader, SentenceDataset
from flair.training_utils import Result, store_embeddings

log = logging.getLogger("flair")


class Lemmatizer(flair.nn.Model):

    def __init__(self,
                 input_size: int = 30,
                 hidden_size: int = 128,
                 num_layers_in_rnn: int = 1,
                 beam_size: int = 5,
                 # path_to_corpus_for_creating_char_dict: str = None,
                 label_type: str = 'lemma',
                 max_sequence_length_dependent_on_input: bool = True,
                 max_sequence_length: int = 20,
                 # path_to_char_dict: str = None,
                 token_embedding: flair.embeddings.TokenEmbeddings = None,
                 use_attention: bool = True,
                 char_dictionary: Union[str, Dictionary] = "common-chars",
                 ):
        """
        Initializes a Lemmatizer model
        The model consists of a decoder and an encoder. The decoder is a Token-Embedding from flair. The embedding of a token
        is handed as the initial hidden state to the decoder, which is a RNN-cell (torch.nn.GRU) that predicts the lemma of
        the given token one letter at a time.
        Note that for learning one can use data in which only those words are annotated that differ from their lemma or data in which
        all words are annotated with a (maybe equal) lemma.
        :param token_embedding: Embedding used to encode sentence
        :param input_size: Input size of the RNN's. Each letter of a token is represented by a hot-one-vector over the given character
            dictionary. This vector is transformed to a input_size vector with a linear layer.
        :param hidden_size: size of the hidden state of the RNN. The initial embedding is transformed to a vector of size hidden_size
            with an additional linear layer
        :param num_layers_in_rnn: Number of stacked RNN cells
        :param path_to_char_dict: Path to character dictionary. If None, a standard dictionary will be used.
            Note that there are some rules to the dictionary. The first three indices must be reserved for the special characters
            <>, <S>, <E>, in that order. The characters represent a dummy symbol, start and end of a word, respectively.
        :param label_type: Name of the gold labels to use.
        :param beam_size: Number of hypothesis used when decoding the output of the RNN. Only used in prediction.
        :param max_sequence_length_dependent_on_input: If set to True, the maximum length of a decoded sequence in the prediction
            depends on the sentences you want to lemmatize. To be precise the maximum length is computed as the length of the longest
            token in the sentences plus one.
        :param max_sequence_length: If set to True and max_sequence_length_dependend_on_input is False a fixed maximum length for
            the decoding will be used for all sentences.
        """

        super().__init__()  # TODO: LSTMs instead of GRUs, maybe as a parameter?

        self._label_type = label_type
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.dependend_on_input = max_sequence_length_dependent_on_input
        self.use_attention = use_attention

        # encoder: embed tokens and map embedding to hidden size of RNN
        self.token_embedding = token_embedding
        self.hidden_size = hidden_size
        if self.token_embedding:
            self.emb_to_hidden = nn.Linear(token_embedding.embedding_length, hidden_size)

        # decoder: dictionary of characters to create one-hot embeddings for decoder inputs
        self.char_dictionary = char_dictionary if type(char_dictionary) is Dictionary else Dictionary(char_dictionary)
        # always add default characters
        self.char_dictionary.add_item('<>')  # index 1
        self.char_dictionary.add_item('<S>')  # index 2
        self.char_dictionary.add_item('<E>')  # index 3

        self.alphabet_size = len(self.char_dictionary)

        # decoder: linear layers to transform vectors to and from alphabet_size
        self.decoder_character_embedding = nn.Embedding(self.alphabet_size, input_size)
        if self.use_attention and token_embedding is None:
            # when using attention we concatenate attention outcome and decoder hidden states
            self.character_decoder = nn.Linear(2 * self.hidden_size, self.alphabet_size)
        else:
            self.character_decoder = nn.Linear(self.hidden_size, self.alphabet_size)

        # decoder RNN
        self.input_size = input_size
        self.num_layers = num_layers_in_rnn

        self.decoder_rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, batch_first=True,
                                  num_layers=num_layers_in_rnn)

        # alternative encoder
        self.encoder_rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, batch_first=True,
                                  num_layers=num_layers_in_rnn)

        self.encoder_character_embedding = nn.Embedding(self.alphabet_size, input_size)

        # loss and softmax
        self.loss = nn.CrossEntropyLoss()
        self.unreduced_loss = nn.CrossEntropyLoss(reduction='none')  # for prediction
        self.softmax = nn.Softmax(dim=2)

        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def words_to_char_indices(self, words: List[str], for_input=True, seq_length=None):
        """
        For a given list of labels (lemmas) this function creates index vectors that represent the characters of the single words in the sentence.
        The "batch size" is given by the number of words in the list. Then each word is represented by sequence_length (maximum word length in the
        list + 1)  many indices representing characters in self.char_dict.
        We add one to the maximum_length since we add either a '1' at the beginning of the vectors (representing the start symbol '<S>') if for_input=True
        or a '2' at the end of the vectors (representing the end symbol '<E>') if for_input = False.
        One can manually set the vector length with the parameter seq_length, though the vector length is always at least maximum word length in the
        list + 1
        """
        index_of_start_symbol = self.char_dictionary.get_idx_for_item('<S>')
        index_of_end_symbol = self.char_dictionary.get_idx_for_item('<E>')

        max_length = max(len(label) for label in words) + 1
        if not seq_length:
            # sequence length of each word is equal to max length of a word in the sentence plus one (start character <S>)
            sequence_length = max_length
        else:
            sequence_length = max(seq_length, max_length)
        # batch size is length of sentence
        tensor = torch.ones(len(words), sequence_length, dtype=torch.long).to(flair.device)
        for i in range(len(words)):
            if for_input:
                tensor[i][0] = index_of_start_symbol  # start character <S> at the beginning
                for index, letter in enumerate(words[i]):
                    tensor[i][index + 1] = self.char_dictionary.get_idx_for_item(letter)

            else:
                tensor[i][len(words[i])] = index_of_end_symbol  # end character <E> in the end
                for index, letter in enumerate(words[i]):
                    tensor[i][index] = self.char_dictionary.get_idx_for_item(letter)

        return tensor

    def forward_pass(self, sentences: Union[List[Sentence], Sentence]):
        # pass (list of) sentence(s) through encoder-decoder

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # create list of all tokens of batch, this way we can hand over all sentences at once
        tokens = [token for sentence in sentences for token in sentence]

        if self.token_embedding:  # input using embeddings
            # embedd sentences
            self.token_embedding.embed(sentences)

            # create inital hidden state tensor for batch (num_layers, batch_size, hidden_size)
            initial_hidden_states = self.emb_to_hidden(
                torch.stack(self.num_layers * [torch.stack([token.get_embedding() for token in tokens])]))

        else:  # encode input using encoder RNN

            input_indices = self.words_to_char_indices([token.text for token in tokens], for_input=False)
            input_vectors = self.encoder_character_embedding(input_indices)
            encoding_output, initial_hidden_states = self.encoder_rnn(input_vectors)

        # get labels, if no label provided take the word itself # TODO: zusätzlicher Parameter bzgl label, ob token ohne label einfach übernommen werden sollen
        labels = [token.get_tag(label_type=self._label_type).value if token.get_tag(
            label_type=self._label_type).value else token.text for token in tokens]

        # get char indices for labels of sentence
        # (batch_size, max_sequence_length) batch_size = #words in sentence, max_sequence_length = length of longest label of sentence + 1
        input_indices = self.words_to_char_indices(labels, for_input=True)

        # get char embeddings
        # (batch_size,max_sequence_length,input_size), i.e. replaces char indices with vectors of length input_size
        input_tensor = self.decoder_character_embedding(input_indices)

        # pass batch through rnn
        output, hn = self.decoder_rnn(input_tensor, initial_hidden_states)
        if not self.token_embedding and self.use_attention:
            attention_coefficients = torch.softmax(torch.matmul(encoding_output, torch.transpose(output, 1, 2)), dim=1)

            # take convex combinations of encoder hidden states as new output using the computed attention coefficients
            attention_output = torch.transpose(
                torch.matmul(torch.transpose(encoding_output, 1, 2), attention_coefficients), 1, 2)

            output = torch.cat((output, attention_output), dim=2)

        # transform output to vectors of size self.alphabet_size -> (batch_size, max_sequence_length, alphabet_size)
        output_vectors = self.character_decoder(output)

        return output_vectors, labels

    def _calculate_loss(self, scores, labels):
        # score vector has to have a certain format for (2d-)loss fct (batch_size, alphabet_size, 1, max_seq_length)
        scores_in_correct_format = scores.permute(0, 2, 1).unsqueeze(2)

        # create target vector (batch_size, max_label_seq_length + 1)
        target = self.words_to_char_indices(labels, for_input=False)

        target.unsqueeze_(1)  # (batch_size, 1, max_label_seq_length + 1)

        return self.loss(scores_in_correct_format, target)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        scores, labels = self.forward_pass(sentences)

        return self._calculate_loss(scores, labels)

    def predict(self, sentences: Union[List[Sentence], Sentence],
                label_name='predicted',
                mini_batch_size: int = 16,
                embedding_storage_mode="None",
                return_loss=False,
                print_prediction=False,
                batching_in_rnn: bool = True):
        '''
        Predict lemmas of words for a given (list of) sentence(s).
        :param sentences: sentences to predict
        :param label_name: label name used for predicted lemmas
        :param mini_batch_size: number of tokens that are send through the RNN simultaneously, assuming batching_in_rnn is set to True
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
            you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        :param return_loss: whether or not to compute and return loss. Setting it to True only makes sense if labels are provided
        :print_prediction: If True, lemmatized sentences will be printed in the console.
        :param batching_in_rnn: If False, no batching will take place in RNN Cell. Tokens are processed one at a time.
        '''
        if self.beam_size == 1:  # batching in RNN only works flawlessly for beam size at least 2
            batching_in_rnn = False

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        # max length of the predicted sequences
        if not self.dependend_on_input:
            max_length = self.max_sequence_length
        else:
            max_length = max([len(token.text) + 1 for sentence in sentences for token in sentence])

        # for printing
        line_to_print = ''

        overall_loss = 0
        number_tokens_in_total = 0

        with torch.no_grad():

            if batching_in_rnn:
                dataloader = DataLoader(dataset=SentenceDataset(sentences), batch_size=mini_batch_size)

                for batch in dataloader:

                    # stop if all sentences are empty
                    if not batch:
                        continue

                    # remove previously predicted labels of this type
                    for sentence in batch:
                        for token in sentence:
                            token.remove_labels(label_name)

                    # create list of tokens in batch
                    tokens_in_batch = [token for sentence in batch for token in sentence]
                    number_tokens = len(tokens_in_batch)
                    number_tokens_in_total += number_tokens

                    if self.token_embedding:  # if token embeddings are given, encoding is just the embedding of the words
                        # embedd sentences
                        self.token_embedding.embed(batch)
                        # create inital hidden state tensor for batch (num_layers, batch_size, hidden_size)
                        hidden_states = self.emb_to_hidden(
                            torch.stack(
                                self.num_layers * [torch.stack([token.get_embedding() for token in tokens_in_batch])]))
                    else:  # otherwise encode input using encoder RNN

                        input_indices = self.words_to_char_indices([token.text for token in tokens_in_batch],
                                                                   for_input=False)

                        input_vectors = self.encoder_character_embedding(
                            input_indices)  # TODO: encode Input in reverse?? Maybe as parameter?
                        encoding_output, hidden_states = self.encoder_rnn(input_vectors)

                        batched_encoding_output = torch.stack(self.beam_size * [encoding_output], dim=1).view(
                            self.beam_size * number_tokens, -1, self.hidden_size)

                    # decoding
                    # create input for first pass (batch_size, 1, input_size), first letter is special character <S> (represented by a '2')
                    # sequence length is always set to one in prediction
                    input_indices = 2 * torch.ones(number_tokens, dtype=torch.long).to(flair.device)
                    input_tensor = self.decoder_character_embedding(input_indices).unsqueeze(1)

                    # first pass
                    output, hidden_states = self.decoder_rnn(input_tensor, hidden_states)

                    if not self.token_embedding and self.use_attention:
                        attention_coefficients = torch.softmax(
                            torch.matmul(encoding_output, torch.transpose(output, 1, 2)), dim=1)

                        # take convex combinations of encoder hidden states as new output using the computed attention coefficients
                        attention_output = torch.transpose(
                            torch.matmul(torch.transpose(encoding_output, 1, 2), attention_coefficients), 1, 2)

                        output = torch.cat((output, attention_output), dim=2)

                    output_vectors = self.character_decoder(output)
                    out_probs = self.softmax(output_vectors).squeeze(1)
                    # make sure no dummy symbol <> or start symbol <S> is predicted
                    out_probs[:, 1:3] = -1
                    # pick top beam size many outputs with highest probabilities
                    probabilities, leading_indices = out_probs.topk(self.beam_size, 1)  # max prob along dimension 1
                    # leading_indices and probabilities have size (batch_size, beam_size)

                    if return_loss:
                        # get labels
                        labels = [token.get_tag(label_type=self._label_type).value if token.get_tag(
                            label_type=self._label_type).value else token.text for token in tokens_in_batch]

                        # target vector represents the labels with vectors of indices for characters
                        target = self.words_to_char_indices(labels, for_input=False, seq_length=max_length)

                        losses = self.unreduced_loss(output_vectors.squeeze(1), target[:, 0])
                        losses = torch.stack(self.beam_size * [losses], dim=1).view(-1, 1)
                        # losses are now in the form (beam_size*batch_size,1)
                        # first beam_size many entries belong to first token of the batch, entries from beam_size + 1 until beam_size+beam_size belong to second token, and so on

                    # keep scores of beam_size many hypothesis for each token in the batch
                    scores = torch.log(probabilities).view(-1, 1)  # (beam_size*batch_size,1)

                    # stack all leading indices of all hypothesis and corresponding hidden states in two tensors
                    leading_indices = leading_indices.view(-1,
                                                           1)  # this vector will be send through the RNN in each iteration
                    hidden_states_beam = torch.stack(self.beam_size * [hidden_states], dim=2).view(self.num_layers, -1,
                                                                                                   self.hidden_size)

                    # save sequences so far
                    sequences = torch.tensor([[i.item()] for i in leading_indices], device=flair.device)

                    # keep track of how many hypothesis were completed for each token
                    n_completed = [0 for _ in range(number_tokens)]  # cpu
                    final_candidates = [[] for _ in range(number_tokens)]  # cpu
                    # max_completed = 5

                    for j in range(1, max_length):

                        # forward pass
                        input_tensor = self.decoder_character_embedding(leading_indices)
                        output, hidden_states_beam = self.decoder_rnn(input_tensor, hidden_states_beam)

                        if not self.token_embedding and self.use_attention:
                            attention_coefficients = torch.softmax(
                                torch.matmul(batched_encoding_output, torch.transpose(output, 1, 2)), dim=1)

                            # take convex combinations of encoder hidden states as new output using the computed attention coefficients
                            attention_output = torch.transpose(
                                torch.matmul(torch.transpose(batched_encoding_output, 1, 2), attention_coefficients), 1,
                                2)

                            output = torch.cat((output, attention_output), dim=2)

                        output_vectors = self.character_decoder(output)
                        out_probs = self.softmax(output_vectors)
                        # out_probs have size (beam_size*batch_size, 1, alphabet_size)
                        # make sure no dummy symbol <> or start symbol <S> is predicted
                        out_probs[:, 0, 1:3] = -1
                        # choose beam_size many indices with highest probabilities
                        probabilities, index_candidates = out_probs.topk(self.beam_size, 2)
                        probabilities.squeeze_(1)
                        index_candidates.squeeze_(1)
                        log_probabilities = torch.log(probabilities)
                        # index_candidates have size (beam_size*batch_size, beam_size)

                        # check if an end symbol <E> has been predicted and, in that case, set hypothesis aside
                        for tuple in (index_candidates == 3).nonzero(as_tuple=False):
                            token_number = tuple[0] // self.beam_size  # index of token in in list tokens_in_batch
                            seq = sequences[tuple[0], :]  # hypothesis sequence
                            score = (scores[tuple[0]] + log_probabilities[tuple[0], tuple[1]]) / (
                                    len(seq) + 1)  # hypothesis score
                            loss = 0
                            if return_loss:
                                o = output_vectors[tuple[0], :, :]
                                t = target[token_number, j].unsqueeze(0)
                                loss = (losses[tuple[0], 0] + self.loss(o, t)) / (
                                        len(seq) + 1)  # average loss of output_vectors of sequence

                            final_candidates[token_number].append((seq, score, loss))
                            n_completed[
                                token_number] += 1  # TODO: remove token if number of completed hypothesis exceeds given value

                            # set score of corresponding entry to -inf so it will not be expanded
                            log_probabilities[tuple[0], tuple[1]] = -inf

                        # get leading_indices for next expansion
                        # find highest scoring hypothesis among beam_size*beam_size possible ones for each token

                        # take beam_size many copies of scores vector and add scores of possible new extensions
                        hypothesis_scores = torch.cat(self.beam_size * [scores],
                                                      dim=1) + log_probabilities  # size (beam_size*batch_size, beam_size)
                        # reshape to vector of size (batch_size, beam_size*beam_size), each row contains beam_size*beam_size scores of the new possible hypothesis
                        hypothesis_scores_per_token = hypothesis_scores.view(number_tokens, self.beam_size ** 2)
                        # choose beam_size best for each token
                        best_scores, indices_per_token = hypothesis_scores_per_token.topk(self.beam_size,
                                                                                          1)  # size (batch_size, beam_size)

                        # out of indices_per_token we now need to recompute the original indices of the hypothesis in a list of length beam_size*batch_size
                        # where the first three inidices belomng to the first token, the next three to the second token, and so on
                        beam_numbers = []
                        seq_numbers = []

                        for i, row in enumerate(indices_per_token):
                            for l, index in enumerate(row):
                                beam = i * self.beam_size + index // self.beam_size
                                seq_number = index % self.beam_size

                                beam_numbers.append(beam.item())
                                seq_numbers.append(seq_number.item())

                        # with these indices we can compute the tensors for the next iteration
                        sequences = torch.cat(
                            (sequences[beam_numbers], index_candidates[beam_numbers, seq_numbers].unsqueeze(1)),
                            dim=1)  # expand sequences with corresponding index
                        scores = scores[beam_numbers] + log_probabilities[beam_numbers, seq_numbers].unsqueeze(
                            1)  # add log-probabilities to the scores
                        leading_indices = index_candidates[beam_numbers, seq_numbers].unsqueeze(
                            1)  # save new leading indices
                        hidden_states_beam = hidden_states_beam[:, beam_numbers, :]  # save corresponding hidden states

                        if return_loss:
                            losses = losses[beam_numbers] + self.unreduced_loss(output_vectors[beam_numbers, 0, :],
                                                                                torch.stack(
                                                                                    self.beam_size * [target[:, j]],
                                                                                    dim=1).view(-1)).unsqueeze(
                                1)  # compute and update losses

                    # it could happen that no end symbol <E> was predicted for some token in all of the max_length iterations
                    # in that case we append one of the final seuqences without end symbol to the final_candidates
                    best_scores, indices = scores.view(number_tokens, -1).topk(1, 1)

                    for j, (score, index) in enumerate(zip(best_scores.squeeze(1), indices.squeeze(1))):
                        if len(final_candidates[j]) == 0:
                            beam = j * self.beam_size + index.item()
                            loss = 0
                            if return_loss:
                                loss = losses[beam, 0] / max_length
                            final_candidates[j].append((sequences[beam, :], score / max_length, loss))

                    # get best final hypothesis for each token
                    output_sequences = []
                    for l in final_candidates:
                        l_ordered = sorted(l, key=lambda tup: tup[1], reverse=True)
                        output_sequences.append(l_ordered[0])

                    # get characters from index sequences and add predicted label to token
                    for i, seq in enumerate(output_sequences):
                        overall_loss += seq[2]
                        predicted_lemma = ''
                        for idx in seq[0]:
                            predicted_lemma += self.char_dictionary.get_item_for_index(idx)
                        line_to_print += predicted_lemma
                        line_to_print += ' '
                        tokens_in_batch[i].add_tag(tag_type=label_name, tag_value=predicted_lemma)

                    store_embeddings(batch, storage_mode=embedding_storage_mode)

            else:  # no batching in RNN
                # still: embedd sentences batch-wise
                dataloader = DataLoader(dataset=SentenceDataset(sentences), batch_size=mini_batch_size)

                for batch in dataloader:
                    if self.token_embedding:
                        # embed sentence
                        self.token_embedding.embed(batch)

                    # no batches in RNN, prediction for each token
                    for sentence in batch:
                        for token in sentence:

                            number_tokens_in_total += 1
                            # remove previously predicted labels of this type
                            token.remove_labels(label_name)

                            if self.token_embedding:
                                hidden_state = self.emb_to_hidden(torch.stack(
                                    self.num_layers * [token.get_embedding()])).unsqueeze(1)  # size (1, 1, hidden_size)
                            else:  # encode input using encoder RNN

                                input_indices = self.words_to_char_indices([token.text], for_input=False)

                                input_vectors = self.encoder_character_embedding(
                                    input_indices)  # TODO: encode input in reverse?? Maybe as parameter?
                                encoding_output, hidden_state = self.encoder_rnn(input_vectors)

                            # input (batch_size, 1, input_size), first letter is special character <S>
                            input_tensor = self.decoder_character_embedding(
                                torch.tensor([2], device=flair.device)).unsqueeze(1)

                            # first pass
                            output, hidden_state = self.decoder_rnn(input_tensor, hidden_state)

                            if not self.token_embedding and self.use_attention:
                                attention_coefficients = torch.softmax(
                                    torch.matmul(encoding_output, torch.transpose(output, 1, 2)), dim=1)

                                # take convex combinations of encoder hidden states as new output using the computed attention coefficients
                                attention_output = torch.transpose(
                                    torch.matmul(torch.transpose(encoding_output, 1, 2), attention_coefficients), 1, 2)

                                output = torch.cat((output, attention_output), dim=2)

                            output_vectors = self.character_decoder(output)
                            out_probs = self.softmax(output_vectors).squeeze(1)
                            # make sure no dummy symbol <> or start symbol <S> is predicted
                            out_probs[0, 1:3] = -1
                            # take beam size many predictions with highest probabilities
                            probabilities, leading_indices = out_probs.topk(self.beam_size,
                                                                            1)  # max prob along dimension 1
                            log_probabilities = torch.log(probabilities)
                            # leading_indices have size (1, beam_size)

                            loss = 0
                            # get target and compute loss
                            if return_loss:
                                label = token.get_tag(label_type=self._label_type).value if token.get_tag(
                                    label_type=self._label_type).value else token.text

                                target = self.words_to_char_indices([label], for_input=False, seq_length=max_length)

                                loss = self.loss(output_vectors.squeeze(0), target[:, 0])

                            # the list sequences will contain beam_size many hypothesis at each point of the prediction
                            sequences = []
                            # create one candidate hypothesis for each prediction
                            for j in range(self.beam_size):
                                # each candidate is a tuple consisting of the predictions so far, the last hidden state, the score/log probability and the loss
                                prediction_index = leading_indices[0][j].item()
                                prediction_log_probability = log_probabilities[0][j]

                                candidate = [[prediction_index], hidden_state, prediction_log_probability, loss]
                                sequences.append(candidate)

                            # variables needed for further beam search
                            n_completed = 0
                            final_candidates = []

                            # Beam search after the first run
                            for i in range(1, max_length):
                                new_sequences = []

                                # expand each candidate hypothesis in sequences
                                for seq, hid, score, seq_loss in sequences:
                                    # create input vector
                                    input_index = torch.tensor([seq[-1]], device=flair.device)
                                    input_tensor = self.decoder_character_embedding(input_index).unsqueeze(1)

                                    # forward pass
                                    output, hidden_state = self.decoder_rnn(input_tensor, hid)

                                    if not self.token_embedding and self.use_attention:
                                        attention_coefficients = torch.softmax(
                                            torch.matmul(encoding_output, torch.transpose(output, 1, 2)), dim=1)

                                        # take convex combinations of encoder hidden states as new output using the computed attention coefficients
                                        attention_output = torch.transpose(
                                            torch.matmul(torch.transpose(encoding_output, 1, 2),
                                                         attention_coefficients), 1, 2)

                                        output = torch.cat((output, attention_output), dim=2)

                                    output_vectors = self.character_decoder(output)
                                    out_probs = self.softmax(output_vectors).squeeze(1)
                                    # make sure no dummy symbol <> or start symbol <S> is predicted
                                    out_probs[0, 1:3] = -1

                                    new_loss = 0
                                    if return_loss:
                                        new_loss = seq_loss + self.loss(output_vectors.squeeze(0), target[:, i])

                                    # get top beam_size predictions
                                    probabilities, leading_indices = out_probs.topk(self.beam_size,
                                                                                    1)  # max prob along dimension 1
                                    log_probabilities = torch.log(probabilities)

                                    # go through each of the top beam_size predictions
                                    for j in range(self.beam_size):

                                        prediction_index = leading_indices[0][j].item()
                                        prediction_log_probability = log_probabilities[0][j].item()

                                        # add their log probability to previous score
                                        s = score + prediction_log_probability

                                        # if this prediction is a STOP symbol, set it aside
                                        if prediction_index == 3:
                                            candidate = [seq, s / (len(seq) + 1), new_loss / (len(seq) + 1)]
                                            final_candidates.append(candidate)
                                            n_completed += 1
                                        # else, create a new candidate hypothesis with updated score and prediction sequence
                                        else:
                                            candidate = [seq + [prediction_index], hidden_state, s, new_loss]
                                            new_sequences.append(candidate)

                                if len(new_sequences) == 0:  # only possible if self.beam_size is 1 and a <E> was predicted
                                    break

                                # order final candidates by score (in descending order)
                                seq_sorted = sorted(new_sequences, key=lambda tup: tup[2], reverse=True)

                                # only use top beam_size hypothesis as starting point for next iteration
                                sequences = seq_sorted[:self.beam_size]

                            # take one of the beam_size many sequences without predicted <E> symbol, if no end symbol was predicted
                            if len(final_candidates) == 0:
                                seq_without_end = sequences[0]
                                candidate = [seq_without_end[0], seq_without_end[2] / max_length,
                                             seq_without_end[3] / max_length]
                                final_candidates.append(candidate)

                            # order final candidates by score (in descending order)
                            ordered = sorted(final_candidates, key=lambda tup: tup[1], reverse=True)
                            best_sequence = ordered[0]

                            overall_loss += best_sequence[2]

                            # get lemma from indices and add label to token
                            predicted_lemma = ''
                            for idx in best_sequence[0]:
                                predicted_lemma += self.char_dictionary.get_item_for_index(idx)
                            line_to_print += predicted_lemma
                            line_to_print += ' '
                            token.add_tag(tag_type=label_name, tag_value=predicted_lemma)

                        store_embeddings(sentence, storage_mode=embedding_storage_mode)

            if print_prediction:
                print(line_to_print)

            if return_loss:
                return overall_loss, number_tokens_in_total

    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            gold_label_type: str,
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 16,
            num_workers: int = 8,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
            exclude_labels: List[str] = [],
            gold_label_dictionary: Optional[Dictionary] = None,
            batching_in_rnn: bool = True
    ) -> Result:
        """
        This function evaluates a lemmatizer on a given set of sentences
        :param sentences: sentences on which to evaluate
        :param gold_label_type: name of the gold label that is used
        :param out_path: path to the output file in which the predictions of the lemmatizer will be stored
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
            you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        :param mini_batch_size: batch size, also used in prediction
        :param main_evaluation_metric:
        :param exclude_labels:
        :param gold_label_dictionary:
        :param batching_in_rnn:
        """

        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)

        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        with torch.no_grad():

            # loss calculation
            eval_loss = 0
            average_over = 0

            all_labels_and_predictions = []  # list to save sentences as strings and lemma-prediction pairs of token in respective sentence
            all_labels_dict = Dictionary(add_unk=True)  # give lemmas an id
            all_label_names = []  # list to save lemma names

            for batch in data_loader:

                # remove any previously predicted labels
                for sentence in batch:
                    for token in sentence:
                        token.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(batch,
                                              embedding_storage_mode=embedding_storage_mode,
                                              mini_batch_size=mini_batch_size,
                                              label_name='predicted',
                                              return_loss=True,
                                              batching_in_rnn=batching_in_rnn)

                average_over += loss_and_count[1]
                eval_loss += loss_and_count[0]

                # get the gold labels
                for sentence in batch:

                    sentence_labels_and_predictions = []
                    for token in sentence:
                        # get gold label
                        lemma_gold = token.get_labels(gold_label_type)[0].value
                        if not lemma_gold:  # if no lemma is provided take the actual word, assuming only those words with not coiniciding lemma are annotated
                            lemma_gold = token.text
                        # save gold label name and give it an id
                        all_labels_dict.add_item(lemma_gold)
                        all_label_names.append(lemma_gold)
                        # get prediction
                        prediction = token.get_labels('predicted')[0].value
                        # save gold label - prediction pair as tuple
                        sentence_labels_and_predictions.append((lemma_gold, prediction))

                    # save sentence and all pairs of gold labels and predictions in list
                    all_labels_and_predictions.append((sentence.to_plain_string(), sentence_labels_and_predictions))

            # write all original sentences, lemmas and predicted values to out file if given
            if out_path:
                with open(Path(out_path), "w", encoding="utf-8") as outfile:
                    for tuple in all_labels_and_predictions:
                        outfile.write(tuple[0] + '\n')  # the sentence
                        labels = [x[0] for x in tuple[1]]
                        predictions = [x[1] for x in tuple[1]]
                        outfile.write((' ').join(labels) + '\n')
                        outfile.write((' ').join(predictions) + '\n\n')

            # now we come to the evaluation
            y_true = []
            y_pred = []

            # add ids of lemmas and predictions to y_true and y_pred
            # if the prediction did not appear in the gold lemmas it is mapped to the index 0
            for tuple in all_labels_and_predictions:
                for lemma, prediction in tuple[1]:
                    y_true.append(all_labels_dict.get_idx_for_item(lemma))
                    y_pred.append(all_labels_dict.get_idx_for_item(prediction))

            # sort by number of occurrences
            counter = Counter()
            counter.update(all_label_names)

            target_names = []
            corresponding_ids = []

            # go through lemma names in order of frequency
            for label_name, count in counter.most_common():

                if label_name in exclude_labels: continue
                target_names.append(label_name)
                corresponding_ids.append(all_labels_dict.get_idx_for_item(label_name))

            target_names.append('no corresponding gold lemma')
            corresponding_ids.append(0)

            classification = classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=corresponding_ids,
            )

            classification_report_dict = classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True,
                labels=corresponding_ids,
            )

            accuracy = round(accuracy_score(y_true, y_pred), 4)

            try:
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            except KeyError:  # if classification report has no micro avg this means micro avg and accuracy coincide!
                precision_score = accuracy
                recall_score = accuracy
                micro_f_score = accuracy

            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            try:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]
            except:
                main_score = accuracy

            detailed_result = (
                    "\nResults:"
                    f"\n- F-score (micro) {micro_f_score}"
                    f"\n- F-score (macro) {macro_f_score}"
                    f"\n- Accuracy {accuracy}"
                    "\n\nBy class:\n" + classification
            )

            # line for log file
            log_header = "PRECISION\tRECALL\tF1\tACCURACY"
            log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy}"

            if average_over > 0:
                eval_loss /= average_over

            result = Result(
                main_score=main_score,
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
                classification_report=classification_report_dict,
                loss=eval_loss
            )

            return result

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.token_embedding,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "path_to_char_dict": self.path_to_char_dict,
            "num_layers_in_rnn": self.num_layers,
            "label_type": self._label_type,
            "beam_size": self.beam_size,
            "max_sequence_length": self.max_sequence_length,
            "dependend_on_input": self.dependend_on_input,
            "use_attention": self.use_attention
        }

        return model_state

    def _init_model_with_state_dict(state):
        model = Lemmatizer(
            token_embedding=state["embeddings"],
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            path_to_char_dict=state["path_to_char_dict"],
            num_layers_in_rnn=state["num_layers_in_rnn"],
            label_type=state["label_type"],
            beam_size=state["beam_size"],
            max_sequence_length_dependent_on_input=state["dependend_on_input"],
            max_sequence_length=state["max_sequence_length"],
            use_attention=state["use_attention"]
        )
        model.load_state_dict(state["state_dict"])
        return model

    def create_char_dict_from_corpus(corpus: Corpus) -> Dictionary:
        char_dict = Dictionary(add_unk=True)

        char_dict.add_item('<>')  # index 1
        char_dict.add_item('<S>')  # index 2
        char_dict.add_item('<E>')  # index 3

        for sen in corpus.get_all_sentences():
            for token in sen:
                for character in token.text:
                    char_dict.add_item(character)

        return char_dict
