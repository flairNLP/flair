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
from flair.file_utils import cached_path

log = logging.getLogger("flair")

start_token: str = '\t'  # Start-of-sentence token
end_token: str = '\n'  # End-of-sentence token


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
            teacher_forcing_ratio: float = 1,
            longest_word_length: int = 50,
            contextualized_embedding: bool = False
    ):
        """
        :param hidden_size: number of hidden states in RNN
        :param pre_embeddings: pre-trained embeddings
        :param character_dictionary: dictionary containing all the characters in the corpus
        :param n_layers: number of RNN layers
        :param tag_list: list of tags planned to be used. Using tag can improve the accuracy of the model,
        Using tag can improve the accuracy of the model, but the model trained with tag does not work well with data without tag.
        :param tag_dictionary: dictionary of tags you need to use.(The all possible values of the tag in tag_list)
        :param dropout: dictionary contains all the tag values in the tag list
        :param teacher_forcing_ratio: the probability of using teacher_forcing. If it is greater than or equal to 1,
        always use teacher forcing. If less than or equal to 0, never use.
        :param longest_word_length: record the maximum length of words and lemma to use in prediction.
        Based on experience, the default value is set at 50.
        :param contextualized_embedding: Whether to use context-sensitive word embedding. Using context-sensitive word
        embedding can improve the accuracy of the model, but it takes more time for training and prediction.
        """

        super(Lemmatization, self).__init__()
        self.hidden_size = hidden_size
        self.pre_embeddings = pre_embeddings.to(flair.device)
        self.n_layers = n_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
        self.character_dictionary = character_dictionary
        self.character_dictionary.add_item(start_token)
        self.character_dictionary.add_item(end_token)

        self.contextualized_embedding = contextualized_embedding
        if not contextualized_embedding:
            # Classic word embedding. Loading weights from a pre-trained word embedding model.
            self.pre_embedding_weight = self._load_embedding_weight(self.character_dictionary, self.pre_embeddings)
            self.embedding = nn.Embedding(len(self.character_dictionary), self.pre_embeddings.embedding_length)
            self.embedding.weight.data.copy_(self.pre_embedding_weight)
            self.embedding.weight.requires_grad = False
        else:
            # Contextualized word embedding.
            representation = self.pre_embeddings.lm.get_representation(" ", start_token, end_token)
            # Get the expression of the start sign in the contextualized word embedding.
            self.start_representation = representation[:len(start_token)].detach()
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

        # Use tag only when tag_list and tag_dictionary are not None, and tag_list is not empty.
        if tag_list is not None and tag_dictionary is not None:
            if len(tag_list) > 0:
                self.use_tag: bool = True
                self.tag_list = tag_list
                self.tag_dictionary = tag_dictionary
                self.tag_embeddings = nn.Embedding(len(self.tag_dictionary), self.pre_embeddings.embedding_length)
                self.tag_embeddings.weight.data.uniform_(-1, 1)
            else:
                self.use_tag: bool = False
                log.warning("tag_list is empty")
        elif tag_list is None and tag_dictionary is None:
            self.use_tag: bool = False
        else:
            self.use_tag: bool = False
            log.warning("If you want to use tag when training the model, do not forget either of the tag_list and tag_dictionary parameters.")

        # Record the maximum length of words and lemma to use in prediction
        self.longest_word_length: int = longest_word_length

        self.to(flair.device)


    def _load_embedding_weight(self, dictionary: Dictionary, pre_embedding: FlairEmbeddings):
        """
        According to character_dictionary, load weights from pre-trained Embedding.
        When using classical word embedding, use this method to load the weights of the embedding layer.
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
            encoder_input, lemma, mask, effective_data_lenght, max_lemma_lenght = self._generate_input(sentence)

            encoder_out, encoder_hidden = self._encode(encoder_input,  effective_data_lenght)

            decoder_hidden = encoder_hidden[:self.n_layers]

            #Using a contextualized word embedding model, it is necessary to generate a representation for the current moment
            # based on the previously predicted results and context. The dictionary all_seqs is used to store the previous prediction results.
            if self.contextualized_embedding:
                all_seqs = dict()
                for i in range(len(effective_data_lenght)):
                    all_seqs[i] = ""

            #If you use teacher forcing, the next input uses the correct value, otherwise use the predicted value as input for the next moment.
            # The probability of using Teacher Forcing can be modified by adjusting the value of teacher_forcing_ratio.
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            decoder_input = None
            if use_teacher_forcing:
                for t in range(max_lemma_lenght):
                    if self.contextualized_embedding:
                            #The first input of the decoder uses the start symbol. Afterwards, the result of the previous
                            # moment is used as the input value for the next moment.
                            if decoder_input == None:
                                decoder_input = self.start_representation.repeat(1, len(all_seqs), 1)
                            else:
                                decoder_input = self.pre_embeddings.lm.get_representation(list(all_seqs.values()), "", "")[-1:].detach()
                    else:
                        if decoder_input == None:
                            start_idx = self.character_dictionary.get_idx_for_item(start_token)
                            decoder_input = torch.LongTensor([[start_idx for _ in range(len(effective_data_lenght))]]).to(flair.device)
                            decoder_input = self.embedding(decoder_input)
                        else:
                            decoder_input = self.embedding(decoder_input)

                    decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_out)
                    mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
                    loss += mask_loss
                    # Use the correct value as input for the next moment
                    decoder_input = lemma[t].view(1, -1)

                    # Using contextualized word embeddings requires recording the prediction results at each moment to generate word vectors.
                    if self.contextualized_embedding:
                        for i in range(len(effective_data_lenght)):
                            all_seqs[i] += self.character_dictionary.get_item_for_index(decoder_input[0][i])

            else:
                for t in range(max_lemma_lenght):
                    if self.contextualized_embedding:
                        if decoder_input == None:
                            decoder_input = self.start_representation.repeat(1, len(all_seqs), 1)
                        else:
                            decoder_input = self.pre_embeddings.lm.get_representation(list(all_seqs.values()), "", "")[-1:].detach()
                    else:
                        if decoder_input == None:
                            start_idx = self.character_dictionary.get_idx_for_item(start_token)
                            decoder_input = torch.LongTensor([[start_idx for _ in range(len(effective_data_lenght))]]).to(flair.device)
                            decoder_input = self.embedding(decoder_input)
                        else:
                            decoder_input = self.embedding(decoder_input)

                    decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_out)
                    mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
                    loss += mask_loss

                    # Use the value with the highest probability as the predicted value, i.e. the input for the next moment.
                    _, topi = decoder_output.topk(1)

                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(len(effective_data_lenght))]]).to(flair.device)

                    if self.contextualized_embedding:
                        for i in range(len(effective_data_lenght)):
                            all_seqs[i] += self.character_dictionary.get_item_for_index(decoder_input[0][i])


        return loss

    def _encode(self, encoder_input, effective_data_lenght):
        """
        :param encoder_input: the output of the word embedding layer, i.e. the word vector,
        :param effective_data_lenght: the valid length of each column.i.e. the original length of each token.
        """
        hidden = None
        packed = nn.utils.rnn.pack_padded_sequence(encoder_input, effective_data_lenght)
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

    def _decode(self, decode_input, last_hidden, encoder_outputs):

        decode_input = self.embedding_dropout(decode_input)
        rnn_output, hidden = self.decoder(decode_input, last_hidden)
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

        with torch.no_grad():
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

        encoder_input, lemma, mask, effective_data_lenght, max_lemma_lenght = self._generate_input(sentence)

        encoder_outputs, encoder_hidden = self._encode(encoder_input,  effective_data_lenght)
        decoder_hidden = encoder_hidden[:self.n_layers]

        if self.contextualized_embedding:
            all_seqs = dict()
            for i in range(len(effective_data_lenght)):
                all_seqs[i] = ""

        decoder_input = None

        for t in range(max_lemma_lenght):
            if self.contextualized_embedding:
                if decoder_input == None:
                    decoder_input = self.start_representation.repeat(1, len(all_seqs), 1)
                else:
                    decoder_input = self.pre_embeddings.lm.get_representation(list(all_seqs.values()), "", "")[-1:].detach()
            else:
                if decoder_input == None:
                    start_idx = self.character_dictionary.get_idx_for_item(start_token)
                    decoder_input = torch.LongTensor([[start_idx for _ in range(len(effective_data_lenght))]]).to(flair.device)
                    decoder_input = self.embedding(decoder_input)
                else:
                    decoder_input = self.embedding(decoder_input)

            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_outputs)

            # Take the predicted value with the highest probability as the input for the next moment
            _, topi = decoder_output.topk(1)

            mask_loss = self._calculate_loss(decoder_output, lemma[t], mask[t])
            loss += mask_loss
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(len(effective_data_lenght))]]).to(flair.device)

            if self.contextualized_embedding:
                for i in range(len(effective_data_lenght)):
                    all_seqs[i] += self.character_dictionary.get_item_for_index(decoder_input[0][i])

        return loss

    def _eval_predict(self, sentence: Sentence):

        encoder_input, _, _, effective_data_lenght, _ = self._generate_input(sentence)

        pre_lemmas = self._calculation_output(encoder_input, effective_data_lenght)

        return  pre_lemmas

    def _calculation_output(self, encoder_input, effective_data_lenght):

        # Calculate the output of the model based on the input, use it in the prediction.
        encoder_outputs, encoder_hidden = self._encode(encoder_input, effective_data_lenght)
        decoder_hidden = encoder_hidden[:self.n_layers]

        end_idx = self.character_dictionary.get_idx_for_item(end_token)

        # Use all_seqs dictionary to save the results
        all_seqs = dict()
        for i in range(len(effective_data_lenght)):
            all_seqs[i] = ""

        decoder_input = None
        for t in range(self.longest_word_length):
            all_seq_is_end = True
            if self.contextualized_embedding:
                if decoder_input == None:
                    decoder_input = self.start_representation.repeat(1, len(all_seqs), 1)
                else:
                    decoder_input = self.pre_embeddings.lm.get_representation(list(all_seqs.values()), "", "")[-1:].detach()
            else:
                if decoder_input == None:
                    start_idx = self.character_dictionary.get_idx_for_item(start_token)
                    decoder_input = torch.LongTensor([[start_idx for _ in range(len(effective_data_lenght))]]).to(flair.device)
                    decoder_input = self.embedding(decoder_input)
                else:
                    decoder_input = self.embedding(decoder_input)

            decoder_output, decoder_hidden = self._decode(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            decoder_input = torch.unsqueeze(decoder_input, 0)

            for i in range(len(effective_data_lenght)):
                all_seqs[i] += self.character_dictionary.get_item_for_index(decoder_input[0][i])
                # If there is a value that is not a end_token in the result, the prediction will continue
                if  decoder_input[0][i] != end_idx:
                    all_seq_is_end = False
            # To improve efficiency, end the prediction early when all predictions result in a end_token.
            if all_seq_is_end:
                break

        # Formatting the results
        pre_lemmas = self._processing_output(all_seqs)

        return pre_lemmas

    def _processing_output(self, pre_lemmas: dict):

        #Process the predicted sequence by splitting the sequence from the first end_token.
        lemmas = list(pre_lemmas.values())

        for i in range(len(lemmas)):
            lemmas[i] = lemmas[i].split('\n')[0]

        return lemmas

    def predict(self, sentence: Sentence,  set_label: bool = True):

        # Predict the lemma of each token in the sentence
        if type(sentence) ==  Sentence:

            pre_lemmas  = self._eval_predict(sentence)

            # The order of the predicted results obtained by our prediction function is sorted, so we need to sort
            # the tokens in the sentence to match the predicted value
            tokens = sentence.tokens[:]
            tokens.sort(key=lambda x: len(x.text), reverse=True)

            # The lemmas list stores lemmas in the order of the tokens in the original sentence
            lemmas = []
            for _ in range(len(sentence)):
                lemmas.append("")
            # Use the idx of the token to match the order of the token in the original sentence
            for i in range(len(tokens)):
                if set_label:
                    sentence[tokens[i].idx - 1].set_label("lemma", pre_lemmas[i])

                lemmas[tokens[i].idx - 1] =  pre_lemmas[i]

            return lemmas
        else:
            log.warning("The acceptable input type is Sentence.")

            return ''


    def _generate_input(self, sentence: Sentence):
        """
        Preprocess data, generate word vector representation of data. Also generate the sequence of lemma based on the
        character dictionary.
        Return tensor. Due to nn.utils.rnn.pack_padded_sequence, the sequence is arranged from long to short.
        :return: embed_data, lemma_seq_list, mask, effective_data_lenght, max_lemma_lenght
        example: Sentence("I am cooking. .", use_tokenizer=False)
        char_dictionary: {6: "a", 7: "m", 8: "n", 10: "e", 12: "i", 13: "c", 15: "o" , 17: "k", 21: "b", 30: ".", 33:"g", 37: "I", 116: end_token}
        embed_data: if self.contextualized_embedding is True,
        generate word vectors representation from strings.
        representation = embeddings.lm.get_representation(["I am cooking .", "", ""]
        token_i_representation = representation[0:1]
        token_am_representation = representation[2:4]
        token_cooking_representation = representation[5:12]
        token_period_representation = representation[13:14]
        else use classic word embedding. First generate a sequence of tokens based on the character dictionary.
        token_seq : tensor([[13, 6, 37, 30].  then calculate the word vector through the embedding layer.
                            [15, 7, 0, 0],
                            [15, 0, 0, 0],
                            [17, 0, 0, 0],
                            [12, 0, 0, 0],
                            [8, 0, 0, 0],
                            [33, 0, 0, 0]])

        lemma_seq_list: tensor([[ 13,  21,  37,  30],
                                [ 15,  10, 116, 116],
                                [ 15, 116, 116, 116],
                                [ 17, 116, 116, 116],
                                [116, 116, 116, 116]])
        mask: BoolTensor. The zero-padded part of the sequence should be ignored when calculating the loss
        effective_data_lenght: Record the original length of the input sequence.
        max_lemma_lenght: In this set of data, the maximum length of lemma.(In order to reduce the number of
                        unnecessary iterations during training)
        """

        # Process the input data and sort the tokens in the sentence from longest to shortest according to their length.
        tokens = sentence.tokens[:]
        tokens.sort(key=lambda x: len(x.text), reverse=True)

        effective_data_lenght = []
        lemma_seq_list = []

        if self.use_tag:
            target_lenght = len(tokens[0].text) + len((self.tag_list))
        else:
            target_lenght = len(tokens[0].text)

        max_lemma_lenght = 0

        embed_data = None

        # If use context-sensitive word vectors. Record the position of the character in the string to obtain the word embedding of the required character
        if self.contextualized_embedding:
            representation = self.pre_embeddings.lm.get_representation([sentence.to_original_text()], '', '').detach()
            if tokens[0].start_pos == None:
                token_pos = dict()
                start = 0
                for token in sentence:
                    end = start + len(token.text)
                    token_pos[token.text] = [start, end]
                    start = end + 1

        for token in  tokens:
            if self.contextualized_embedding:
                # If the input sentecen is use_tokenizer, then use its own postion, otherwise use the location dictionary we generated.
                if token.start_pos == None:
                    pos = token_pos[token.text]
                    token_representation = representation[pos[0]:pos[1]]
                else:
                    token_representation = representation[token.start_pos:token.end_pos]
            else:
                word_seq = self._item2seq(self.character_dictionary, token.text)
                word_seq = torch.LongTensor([word_seq]).transpose(0, 1).to(flair.device)
                token_representation = self.embedding(word_seq)

            # If you use tag, add the representation of tag to token_representation
            if self.use_tag:
                tags_seq = []
                for tag in self.tag_list:
                    tags_seq.append(token.get_tag(tag).value)
                tags_seq = self._item2seq(self.tag_dictionary, tags_seq)
                tags_seq = torch.LongTensor([tags_seq]).transpose(0, 1).to(flair.device)
                tag_emb = self.tag_embeddings(tags_seq)

                token_representation = torch.cat((token_representation, tag_emb), dim=0)

            effective_data_lenght.append(len(token_representation))
            pad = torch.zeros(target_lenght - len(token_representation), 1, self.pre_embeddings.embedding_length).to(flair.device)

            # The first token must be the longest, so there is no need to pad.
            if embed_data == None:
                embed_data = token_representation
            else:
                token_representation = torch.cat((token_representation, pad), 0)
                embed_data = torch.cat((embed_data, token_representation), 1)

            lemma_seq = self._item2seq(self.character_dictionary, token.get_tag('lemma').value)
            lemma_seq.append(self.character_dictionary.get_idx_for_item(end_token))

            max_lemma_lenght = max(max_lemma_lenght, len(lemma_seq))
            lemma_seq_list.append(lemma_seq)


        lemma_seq_list = self._seq_zero_padding(lemma_seq_list, max_lemma_lenght, self.character_dictionary.get_idx_for_item(end_token))
        lemma_seq_list = torch.LongTensor(lemma_seq_list).transpose(0, 1).to(flair.device)

        # Generate mask, because when calculating loss, the part through zero padding should not be included in the calculation.
        mask = torch.BoolTensor(self._binary_matrix(lemma_seq_list)).to(flair.device)

        #  update the length of the longest word encountered for the model.
        self.longest_word_length = max(self.longest_word_length, target_lenght, max_lemma_lenght)

        return embed_data, lemma_seq_list, mask, effective_data_lenght, max_lemma_lenght

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
            "contextualized_embedding": self.contextualized_embedding
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
            contextualized_embedding=state["contextualized_embedding"]
        )
        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}

        model_map["lemma-with-pos"] = "https://box.hu-berlin.de/seafhttp/files/41ce986d-8417-4199-b02f-ac153c17cd4b/en-lemma-with-pos.pt"
        model_map["lemma"] = "https://box.hu-berlin.de/seafhttp/files/fa7c2935-bce4-4d6a-9f26-c92ce995660b/en-lemma-without-tag.pt"

        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=Path("models") / "lemma")

        return model_name
