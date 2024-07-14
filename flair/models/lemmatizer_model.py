import logging
from math import inf
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Token
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.training_utils import Result, store_embeddings

log = logging.getLogger("flair")


class Lemmatizer(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        embeddings: Optional[flair.embeddings.TokenEmbeddings] = None,
        label_type: str = "lemma",
        rnn_input_size: int = 50,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 2,
        encode_characters: bool = True,
        char_dict: Union[str, Dictionary] = "common-chars-lemmatizer",
        max_sequence_length_dependent_on_input: bool = True,
        max_sequence_length: int = 20,
        use_attention: bool = True,
        beam_size: int = 1,
        start_symbol_for_encoding: bool = True,
        end_symbol_for_encoding: bool = True,
        bidirectional_encoding: bool = True,
    ) -> None:
        """Initializes a Lemmatizer model.

        The model consists of a decoder and an encoder. The encoder is either a RNN-cell (torch.nn.GRU)
        or a Token-Embedding from flair if an embedding is handed to the constructor (token_embedding).
        The output of the encoder is used as the initial hidden state to the decoder, which is an RNN-cell (GRU)
        that predicts the lemma of the given token one letter at a time.
        Note that one can use data in which only those words are annotated that differ from their lemma or data
        in which all words are annotated with a (maybe equal) lemma.

        Args:
            encode_characters: If True, use a character embedding to additionally encode tokens per character.
            start_symbol_for_encoding: If True, use a start symbol for encoding characters.
            end_symbol_for_encoding: If True, use an end symbol for encoding characters.
            bidirectional_encoding: If True, the character encoding is bidirectional.
            embeddings: Embedding used to encode sentence
            rnn_input_size: Input size of the RNN('s). Each letter of a token is represented by a hot-one-vector over
                the given character dictionary. This vector is transformed to a input_size vector with a linear layer.
            rnn_hidden_size: size of the hidden state of the RNN('s).
            rnn_layers: Number of stacked RNN cells
            beam_size: Number of hypothesis used when decoding the output of the RNN. Only used in prediction.
            char_dict: Dictionary of characters the model is able to process. The dictionary must contain <unk> for
                the handling of unknown characters. If None, a standard dictionary will be loaded. One can either hand
                over a path to a dictionary or the dictionary itself.
            label_type: Name of the gold labels to use.
            max_sequence_length_dependent_on_input: If set to True, the maximum length of a decoded sequence in
                the prediction depends on the sentences you want to lemmatize. To be precise the maximum length is
                computed as the length of the longest token in the sentences plus one.
            max_sequence_length: If set to True and max_sequence_length_dependend_on_input is False a fixed
                maximum length for the decoding will be used for all sentences.
            use_attention: whether to use attention. Only sensible if encoding via RNN
        """
        super().__init__()

        self._label_type = label_type
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.dependent_on_input = max_sequence_length_dependent_on_input
        self.start_symbol = start_symbol_for_encoding
        self.end_symbol = end_symbol_for_encoding
        self.bi_encoding = bidirectional_encoding
        self.rnn_hidden_size = rnn_hidden_size

        # whether to encode characters and whether to use attention (attention can only be used if chars are encoded)
        self.encode_characters = encode_characters
        self.use_attention = use_attention
        if not self.encode_characters:
            self.use_attention = False

        # character dictionary for decoding and encoding
        self.char_dictionary = char_dict if isinstance(char_dict, Dictionary) else Dictionary.load(char_dict)

        # make sure <unk> is in dictionary for handling of unknown characters
        if not self.char_dictionary.add_unk:
            raise KeyError("<unk> must be contained in char_dict")

        # add special symbols to dictionary if necessary and save respective indices
        self.dummy_index = self.char_dictionary.add_item("<>")
        self.start_index = self.char_dictionary.add_item("<S>")
        self.end_index = self.char_dictionary.add_item("<E>")

        # ---- ENCODER ----
        # encoder character embeddings
        self.encoder_character_embedding = nn.Embedding(len(self.char_dictionary), rnn_input_size)

        # encoder pre-trained embeddings
        self.encoder_embeddings = embeddings

        hidden_input_size = 0
        if embeddings:
            hidden_input_size += embeddings.embedding_length
        if encode_characters:
            hidden_input_size += rnn_hidden_size
        if encode_characters and bidirectional_encoding:
            hidden_input_size += rnn_hidden_size
        self.emb_to_hidden = nn.Linear(hidden_input_size, rnn_hidden_size)

        # encoder RNN
        self.encoder_rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            batch_first=True,
            num_layers=rnn_layers,
            bidirectional=self.bi_encoding,
        )

        # additional encoder linear layer if bidirectional encoding
        if self.bi_encoding:
            self.bi_hidden_states_to_hidden_size: Optional[nn.Linear] = nn.Linear(
                2 * self.rnn_hidden_size, self.rnn_hidden_size, bias=False
            )
        else:
            self.bi_hidden_states_to_hidden_size = None

        # ---- DECODER ----
        # decoder: linear layers to transform vectors to and from alphabet_size
        self.decoder_character_embedding = nn.Embedding(len(self.char_dictionary), rnn_input_size)

        # when using attention we concatenate attention outcome and decoder hidden states
        self.character_decoder = nn.Linear(
            2 * self.rnn_hidden_size if self.use_attention else self.rnn_hidden_size,
            len(self.char_dictionary),
        )

        # decoder RNN
        self.rnn_input_size = rnn_input_size
        self.rnn_layers = rnn_layers

        self.decoder_rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            batch_first=True,
            num_layers=rnn_layers,
        )

        # loss and softmax
        self.loss = nn.CrossEntropyLoss(reduction="sum")
        # self.unreduced_loss = nn.CrossEntropyLoss(reduction='none')  # for prediction
        self.softmax = nn.Softmax(dim=2)

        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def words_to_char_indices(
        self,
        tokens: List[str],
        end_symbol=True,
        start_symbol=False,
        padding_in_front=False,
        seq_length=None,
    ):
        """For a given list of strings this function creates index vectors that represent the characters of the strings.

        Each string is represented by sequence_length (maximum string length + entries for special symbol) many
        indices representing characters in self.char_dict.
        One can manually set the vector length with the parameter seq_length, though the vector length is always
        at least maximum string length in the list.

        Args:
            seq_length: the maximum sequence length to use, if None the maximum is taken..
            tokens: the texts of the toekens to encode
            end_symbol: add self.end_index at the end of each representation
            start_symbol: add self.start_index in front of each representation
            padding_in_front: whether to fill up with self.dummy_index in front or in back of strings
        """
        # add additional columns for special symbols if necessary
        c = int(end_symbol) + int(start_symbol)

        max_length = max(len(token) for token in tokens) + c
        sequence_length = max_length if not seq_length else max(seq_length, max_length)

        # initialize with dummy symbols
        tensor = self.dummy_index * torch.ones(len(tokens), sequence_length, dtype=torch.long).to(flair.device)

        for i in range(len(tokens)):
            dif = sequence_length - (len(tokens[i]) + c)
            shift = 0
            if padding_in_front:
                shift += dif
            if start_symbol:
                tensor[i][0 + shift] = self.start_index
            if end_symbol:
                tensor[i][len(tokens[i]) + int(start_symbol) + shift] = self.end_index
            for index, letter in enumerate(tokens[i]):
                tensor[i][index + int(start_symbol) + shift] = self.char_dictionary.get_idx_for_item(letter)

        return tensor

    def forward_pass(self, sentences: Union[List[Sentence], Sentence]):
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # encode inputs
        initial_hidden_states, all_encoder_outputs = self.encode(sentences)

        # get labels (we assume each token has a lemma label)
        labels = [token.get_label(self._label_type).value for sentence in sentences for token in sentence]

        # get char indices for labels of sentence
        # (batch_size, max_sequence_length) batch_size = #words in sentence,
        # max_sequence_length = length of longest label of sentence + 1
        decoder_input_indices = self.words_to_char_indices(
            labels, start_symbol=True, end_symbol=False, padding_in_front=False
        )

        # get char embeddings
        # (batch_size,max_sequence_length,input_size), i.e. replaces char indices with vectors of length input_size
        output_vectors, _ = self.decode(decoder_input_indices, initial_hidden_states, all_encoder_outputs)

        return output_vectors, labels

    def decode(self, decoder_input_indices, initial_hidden_states, all_encoder_outputs):
        # take decoder input and initial hidden and pass through RNN
        input_tensor = self.decoder_character_embedding(decoder_input_indices)
        output, hidden = self.decoder_rnn(input_tensor, initial_hidden_states)

        # if all encoder outputs are provided, use attention
        if self.use_attention:
            attention_coeff = torch.softmax(torch.matmul(all_encoder_outputs, torch.transpose(output, 1, 2)), dim=1)

            # take convex combinations of encoder hidden states as new output using the computed attention coefficients
            attention_output = torch.transpose(
                torch.matmul(torch.transpose(all_encoder_outputs, 1, 2), attention_coeff),
                1,
                2,
            )

            output = torch.cat((output, attention_output), dim=2)

        # transform output to vectors of size len(char_dict) -> (batch_size, max_sequence_length, alphabet_size)
        output_vectors = self.character_decoder(output)
        return output_vectors, hidden

    def _prepare_tensors(self, sentences: List[Sentence]) -> Tuple[Optional[torch.Tensor], ...]:
        # get all tokens
        tokens = [token for sentence in sentences for token in sentence]

        # encode input characters by sending them through RNN
        if self.encode_characters:
            # get one-hots for characters and add special symbols / padding
            encoder_input_indices = self.words_to_char_indices(
                [token.text for token in tokens],
                start_symbol=self.start_symbol,
                end_symbol=self.end_symbol,
                padding_in_front=False,
            )

            # determine length of each token
            extra = 0
            if self.start_symbol:
                extra += 1
            if self.end_symbol:
                extra += 1
            lengths = torch.tensor([len(token.text) + extra for token in tokens], device=flair.device)
        else:
            encoder_input_indices = None
            lengths = None

        if self.encoder_embeddings:
            # embed sentences
            self.encoder_embeddings.embed(sentences)

            # create initial hidden state tensor for batch (num_layers, batch_size, hidden_size)
            token_embedding_hidden = torch.stack(
                self.rnn_layers * [torch.stack([token.get_embedding() for token in tokens])]
            )
        else:
            token_embedding_hidden = None

        return encoder_input_indices, lengths, token_embedding_hidden

    def forward(
        self,
        encoder_input_indices: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        token_embedding_hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # variable to store initial hidden states for decoder
        initial_hidden_for_decoder = []

        # encode input characters by sending them through RNN
        if encoder_input_indices is not None and lengths is not None:
            input_vectors = self.encoder_character_embedding(encoder_input_indices)

            # test packing and padding
            packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                input_vectors,
                lengths,
                enforce_sorted=False,
                batch_first=True,
            )

            encoding_flat, initial_hidden_states = self.encoder_rnn(packed_sequence)
            encoder_outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(encoding_flat, batch_first=True)

            # since bidirectional rnn is only used in encoding we need to project outputs to hidden_size of decoder
            if self.bi_encoding and self.bi_hidden_states_to_hidden_size is not None:
                encoder_outputs = self.bi_hidden_states_to_hidden_size(encoder_outputs)

                # concatenate the final hidden states of the encoder. These will be projected to hidden_size of
                # decoder later with self.emb_to_hidden
                conditions = torch.cat(2 * [torch.eye(self.rnn_layers).bool()])
                bi_states = [initial_hidden_states[conditions[:, i], :, :] for i in range(self.rnn_layers)]
                initial_hidden_states = torch.stack([torch.cat((b[0, :, :], b[1, :, :]), dim=1) for b in bi_states])

            initial_hidden_for_decoder.append(initial_hidden_states)

            # mask out vectors that correspond to a dummy symbol (TODO: check attention masking)
            mask = torch.cat(
                (self.rnn_hidden_size * [(encoder_input_indices == self.dummy_index).unsqueeze(2)]),
                dim=2,
            )
            all_encoder_outputs: Optional[torch.Tensor] = torch.where(
                mask, torch.tensor(0.0, device=flair.device), encoder_outputs
            )
        else:
            all_encoder_outputs = None
        # use token embedding as initial hidden state for decoder
        if token_embedding_hidden is not None:
            initial_hidden_for_decoder.append(token_embedding_hidden)

        # concatenate everything together and project to appropriate size for decoder
        initial_hidden = self.emb_to_hidden(torch.cat(initial_hidden_for_decoder, dim=2))

        return initial_hidden, all_encoder_outputs

    def encode(self, sentences: List[Sentence]):
        tensors = self._prepare_tensors(sentences)
        return self.forward(*tensors)

    def encode_token(self, token: Token):
        # variable to store initial hidden states for decoder
        initial_hidden_for_decoder = []
        all_encoder_outputs = None

        # encode input characters by sending them through RNN
        if self.encode_characters:
            # note that we do not need to fill up with dummy symbols since we process each token seperately
            encoder_input_indices = self.words_to_char_indices(
                [token.text], start_symbol=self.start_symbol, end_symbol=self.end_symbol
            )
            # embed character one-hots
            input_vector = self.encoder_character_embedding(encoder_input_indices)
            # send through encoder RNN (produces initial hidden for decoder)
            all_encoder_outputs, initial_hidden_states = self.encoder_rnn(input_vector)

            # since bidirectional rnn is only used in encoding we need to project outputs to hidden_size of decoder
            if self.bi_encoding and self.bi_hidden_states_to_hidden_size is not None:
                # project 2*hidden_size to hidden_size
                all_encoder_outputs = self.bi_hidden_states_to_hidden_size(all_encoder_outputs)

                # concatenate the final hidden states of the encoder. These will be projected to hidden_size of decoder
                # later with self.emb_to_hidden
                conditions = torch.cat(2 * [torch.eye(self.rnn_layers).bool()])
                bi_states = [initial_hidden_states[conditions[:, i], :, :] for i in range(self.rnn_layers)]
                initial_hidden_states = torch.stack([torch.cat((b[0, :, :], b[1, :, :]), dim=1) for b in bi_states])

            initial_hidden_for_decoder.append(initial_hidden_states)

        # use token embedding as initial hidden state for decoder
        if self.encoder_embeddings:
            # create initial hidden state tensor for batch (num_layers, batch_size, hidden_size)
            token_embedding_hidden = torch.stack(self.rnn_layers * [token.get_embedding()]).unsqueeze(1)

            initial_hidden_for_decoder.append(token_embedding_hidden)

        # concatenate everything together and project to appropriate size for decoder
        initial_hidden_for_decoder = self.emb_to_hidden(torch.cat(initial_hidden_for_decoder, dim=2))

        return initial_hidden_for_decoder, all_encoder_outputs

    def _calculate_loss(self, scores, labels):
        # score vector has to have a certain format for (2d-)loss fct (batch_size, alphabet_size, 1, max_seq_length)
        scores_in_correct_format = scores.permute(0, 2, 1).unsqueeze(2)

        # create target vector (batch_size, max_label_seq_length + 1)
        target = self.words_to_char_indices(labels, start_symbol=False, end_symbol=True, padding_in_front=False)

        target.unsqueeze_(1)  # (batch_size, 1, max_label_seq_length + 1)

        return self.loss(scores_in_correct_format, target), len(labels)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, int]:
        scores, labels = self.forward_pass(sentences)

        return self._calculate_loss(scores, labels)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 16,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name="predicted",
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """Predict lemmas of words for a given (list of) sentence(s).

        Args:
            sentences: sentences to predict
            label_name: label name used for predicted lemmas
            mini_batch_size: number of tokens that are send through the RNN simultaneously, assuming batching_in_rnn is set to True
            embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
            return_loss: whether to compute and return loss. Setting it to True only makes sense if labels are provided
            verbose: If True, lemmatized sentences will be printed in the console.
            return_probabilities_for_all_classes: unused parameter.
        """
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        Sentence.set_context_for_sentences(sentences)

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        # max length of the predicted sequences
        if not self.dependent_on_input:
            max_length = self.max_sequence_length
        else:
            max_length = max([len(token.text) + 1 for sentence in sentences for token in sentence])

        # for printing
        line_to_print = ""

        overall_loss = 0.0
        number_tokens_in_total = 0

        with torch.no_grad():
            dataloader = DataLoader(dataset=FlairDatapointDataset(sentences), batch_size=mini_batch_size)

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

                # encode inputs
                hidden, all_encoder_outputs = self.encode(batch)

                # create input for first pass (batch_size, 1, input_size), first letter is special character <S>
                # sequence length is always set to one in prediction
                input_indices = self.start_index * torch.ones(
                    number_tokens, dtype=torch.long, device=flair.device
                ).unsqueeze(1)

                # option 1: greedy decoding
                if self.beam_size == 1:
                    # predictions
                    predicted: List[List[Union[int, float]]] = [[] for _ in range(number_tokens)]

                    for _decode_step in range(max_length):
                        # decode next character
                        output_vectors, hidden = self.decode(input_indices, hidden, all_encoder_outputs)

                        log_softmax_probs = torch.nn.functional.log_softmax(output_vectors, dim=2)
                        # pick top beam size many outputs with highest probabilities
                        input_indices = log_softmax_probs.argmax(dim=2)

                        for i in range(number_tokens):
                            if len(predicted[i]) > 0 and predicted[i][-1] == self.end_index:
                                continue
                            predicted[i].append(input_indices[i].item())

                    for t_id, token in enumerate(tokens_in_batch):
                        predicted_lemma = "".join(
                            self.char_dictionary.get_item_for_index(idx) if idx != self.end_index else ""
                            for idx in predicted[t_id]
                        )
                        token.set_label(typename=label_name, value=predicted_lemma)

                # option 2: beam search
                else:
                    output_vectors, hidden = self.decode(input_indices, hidden, all_encoder_outputs)

                    # out_probs = self.softmax(output_vectors).squeeze(1)
                    log_softmax_probs = torch.nn.functional.log_softmax(output_vectors, dim=2).squeeze(1)
                    # make sure no dummy symbol <> or start symbol <S> is predicted
                    log_softmax_probs[:, self.dummy_index] = -inf
                    log_softmax_probs[:, self.start_index] = -inf

                    # pick top beam size many outputs with highest probabilities
                    # probabilities, leading_indices = out_probs.topk(self.beam_size, 1)  # max prob along dimension 1
                    log_probabilities, leading_indices = log_softmax_probs.topk(self.beam_size, 1)
                    # leading_indices and probabilities have size (batch_size, beam_size)

                    # keep scores of beam_size many hypothesis for each token in the batch
                    scores = log_probabilities.view(-1, 1)
                    # stack all leading indices of all hypothesis and corresponding hidden states in two tensors
                    leading_indices = leading_indices.view(-1, 1)  # this vector goes through RNN in each iteration

                    hidden_states_beam = torch.stack(self.beam_size * [hidden], dim=2).view(
                        self.rnn_layers, -1, self.rnn_hidden_size
                    )

                    # save sequences so far
                    sequences = torch.tensor([[i.item()] for i in leading_indices], device=flair.device)

                    # keep track of how many hypothesis were completed for each token
                    n_completed = [0 for _ in range(number_tokens)]  # cpu
                    final_candidates: List[List[Tuple[torch.Tensor, float]]] = [[] for _ in range(number_tokens)]  # cpu

                    # if all_encoder_outputs returned, expand them to beam size (otherwise keep this as None)
                    batched_encoding_output = (
                        torch.stack(self.beam_size * [all_encoder_outputs], dim=1).view(
                            self.beam_size * number_tokens, -1, self.rnn_hidden_size
                        )
                        if self.use_attention
                        else None
                    )

                    for _j in range(1, max_length):
                        output_vectors, hidden_states_beam = self.decode(
                            leading_indices, hidden_states_beam, batched_encoding_output
                        )

                        # decode with log softmax
                        out_log_probs = torch.nn.functional.log_softmax(output_vectors, dim=2)
                        # make sure no dummy symbol <> or start symbol <S> is predicted
                        out_log_probs[:, 0, self.dummy_index] = -inf
                        out_log_probs[:, 0, self.start_index] = -inf
                        log_probabilities, index_candidates = out_log_probs.topk(self.beam_size, 2)
                        log_probabilities.squeeze_(1)
                        index_candidates.squeeze_(1)

                        # check if an end symbol <E> has been predicted and, in that case, set hypothesis aside
                        end_symbols = (index_candidates == self.end_index).nonzero(as_tuple=False)
                        for tuple in end_symbols:
                            # if the sequence is already ended, do not record as candidate
                            if sequences[tuple[0], -1].item() == self.end_index:
                                continue

                            # index of token in in list tokens_in_batch
                            token_number = torch.div(tuple[0], self.beam_size, rounding_mode="trunc")
                            # print(token_number)
                            seq = sequences[tuple[0], :]  # hypothesis sequence
                            # hypothesis score
                            score = (scores[tuple[0]] + log_probabilities[tuple[0], tuple[1]]) / (len(seq) + 1)

                            final_candidates[token_number].append((seq, score.item()))
                            # TODO: remove token if number of completed hypothesis exceeds given value
                            n_completed[token_number] += 1

                            # set score of corresponding entry to -inf so it will not be expanded
                            log_probabilities[tuple[0], tuple[1]] = -inf

                        # get leading_indices for next expansion
                        # find highest scoring hypothesis among beam_size*beam_size possible ones for each token

                        # take beam_size many copies of scores vector and add scores of possible new extensions
                        # size (beam_size*batch_size, beam_size)
                        hypothesis_scores = torch.cat(self.beam_size * [scores], dim=1) + log_probabilities
                        # print(hypothesis_scores)

                        # reshape to vector of size (batch_size, beam_size*beam_size),
                        # each row contains beam_size*beam_size scores of the new possible hypothesis
                        hypothesis_scores_per_token = hypothesis_scores.view(number_tokens, self.beam_size**2)
                        # print(hypothesis_scores_per_token)

                        # choose beam_size best for each token - size (batch_size, beam_size)
                        (
                            best_scores,
                            indices_per_token,
                        ) = hypothesis_scores_per_token.topk(self.beam_size, 1)

                        # out of indices_per_token we now need to recompute the original indices of the hypothesis in
                        # a list of length beam_size*batch_size
                        # where the first three inidices belong to the first token, the next three to the second token,
                        # and so on
                        beam_numbers: List[int] = []
                        seq_numbers: List[int] = []

                        for i, row in enumerate(indices_per_token):
                            beam_numbers.extend(i * self.beam_size + index.item() // self.beam_size for index in row)

                            seq_numbers.extend(index.item() % self.beam_size for index in row)

                        # with these indices we can compute the tensors for the next iteration
                        # expand sequences with corresponding index
                        sequences = torch.cat(
                            (
                                sequences[beam_numbers],
                                index_candidates[beam_numbers, seq_numbers].unsqueeze(1),
                            ),
                            dim=1,
                        )

                        # add log-probabilities to the scores
                        scores = scores[beam_numbers] + log_probabilities[beam_numbers, seq_numbers].unsqueeze(1)

                        # save new leading indices
                        leading_indices = index_candidates[beam_numbers, seq_numbers].unsqueeze(1)

                        # save corresponding hidden states
                        hidden_states_beam = hidden_states_beam[:, beam_numbers, :]

                    # it may happen that no end symbol <E> is predicted for a token in all of the max_length iterations
                    # in that case we append one of the final seuqences without end symbol to the final_candidates
                    best_scores, indices = scores.view(number_tokens, -1).topk(1, 1)

                    for j, (score, index) in enumerate(zip(best_scores.squeeze(1), indices.squeeze(1))):
                        if len(final_candidates[j]) == 0:
                            beam = j * self.beam_size + index.item()
                            final_candidates[j].append((sequences[beam, :], score.item() / max_length))

                    # get best final hypothesis for each token
                    output_sequences = []
                    for candidate in final_candidates:
                        l_ordered = sorted(candidate, key=lambda tup: tup[1], reverse=True)
                        output_sequences.append(l_ordered[0])

                    # get characters from index sequences and add predicted label to token
                    for i, out_seq in enumerate(output_sequences):
                        predicted_lemma = ""
                        for idx in out_seq[0]:
                            predicted_lemma += self.char_dictionary.get_item_for_index(idx)
                        line_to_print += predicted_lemma
                        line_to_print += " "
                        tokens_in_batch[i].add_tag(tag_type=label_name, tag_value=predicted_lemma)

                if return_loss:
                    overall_loss += self.forward_loss(batch)[0].item()

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if verbose:
                log.info(line_to_print)

            if return_loss:
                return overall_loss, number_tokens_in_total
            return None

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.encoder_embeddings.save_embeddings(use_state_dict=False),
            "rnn_input_size": self.rnn_input_size,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_layers": self.rnn_layers,
            "char_dict": self.char_dictionary,
            "label_type": self._label_type,
            "beam_size": self.beam_size,
            "max_sequence_length": self.max_sequence_length,
            "dependent_on_input": self.dependent_on_input,
            "use_attention": self.use_attention,
            "encode_characters": self.encode_characters,
            "start_symbol": self.start_symbol,
            "end_symbol": self.end_symbol,
            "bidirectional_encoding": self.bi_encoding,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            encode_characters=state.get("encode_characters"),
            rnn_input_size=state.get("rnn_input_size"),
            rnn_hidden_size=state.get("rnn_hidden_size"),
            rnn_layers=state.get("rnn_layers"),
            char_dict=state.get("char_dict"),
            label_type=state.get("label_type"),
            beam_size=state.get("beam_size"),
            max_sequence_length_dependent_on_input=state.get("dependent_on_input"),
            max_sequence_length=state.get("max_sequence_length"),
            use_attention=state.get("use_attention"),
            start_symbol_for_encoding=state.get("start_symbol"),
            end_symbol_for_encoding=state.get("end_symbol"),
            bidirectional_encoding=state.get("bidirectional_encoding"),
            **kwargs,
        )

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for sentence in batch:
            eval_line = (
                f" - Text:       {' '.join([token.text for token in sentence])}\n"
                f" - Gold-Lemma: {' '.join([token.get_label(gold_label_type).value for token in sentence])}\n"
                f" - Predicted:  {' '.join([token.get_label('predicted').value for token in sentence])}\n\n"
            )
            lines.append(eval_line)
        return lines

    def evaluate(self, *args, **kwargs) -> Result:
        # Overwrites evaluate of parent class to remove the "by class" printout
        result = super().evaluate(*args, **kwargs)
        result.detailed_results = result.detailed_results.split("\n\n")[0]
        return result
