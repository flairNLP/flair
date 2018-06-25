import torch.autograd as autograd
import torch.nn as nn
import torch
import numpy as np

from flair.file_utils import cached_path
from .data import Dictionary, Sentence, Token
from .embeddings import TextEmbeddings

from typing import List, Tuple

START_TAG: str = '<START>'
STOP_TAG: str = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class SequenceTaggerLSTM(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 embeddings,
                 tag_dictionary: Dictionary,
                 use_crf: bool = True,
                 use_rnn: bool = True,
                 rnn_layers: int = 1
                 ):

        super(SequenceTaggerLSTM, self).__init__()

        self.use_RNN = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tagset_size: int = len(tag_dictionary)

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # self.dropout = nn.Dropout(0.5)
        self.dropout = LockedDropout(0.5)

        rnn_input_dim: int = self.embeddings.embedding_length

        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
            self.embedding2nn = nn.Linear(rnn_input_dim, rnn_input_dim)

        # bidirectional LSTM on top of embedding layer
        self.rnn_type = 'LSTM'
        if self.rnn_type in ['LSTM', 'GRU']:

            if self.nlayers == 1:
                self.rnn = getattr(nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                      num_layers=self.nlayers,
                                                      bidirectional=True)
            else:
                self.rnn = getattr(nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                      num_layers=self.nlayers,
                                                      dropout=0.5,
                                                      bidirectional=True)

        self.relu = nn.ReLU()

        # final linear map to tag space
        if self.use_RNN:
            self.linear = nn.Linear(hidden_size * 2, len(tag_dictionary))
        else:
            self.linear = nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        # trans is also a score tensor, not a probability: THIS THING NEEDS TO GO!!!!
        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.data[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            self.transitions.data[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

    @staticmethod
    def load(model: str):
        model_file = None

        if model.lower() == 'ner':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/ner-conll03.pt'
            model_file = cached_path(base_path)

        if model.lower() == 'chunk':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/chunk-conll2000.pt'
            model_file = cached_path(base_path)

        if model.lower() == 'pos':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/pos-ontonotes-small.pt'
            model_file = cached_path(base_path)

        if model_file is not None:
            tagger: SequenceTaggerLSTM = torch.load(model_file, map_location={'cuda:0': 'cpu'})
            tagger.eval()
            if torch.cuda.is_available():
                tagger = tagger.cuda()
            return tagger

    def forward(self, sentences: List[Sentence], tag_type: str) -> Tuple[List, List]:

        self.zero_grad()

        # first, sort sentences by number of tokens
        sentences.sort(key=lambda x: len(x), reverse=True)
        longest_token_sequence_in_batch: int = len(sentences[0])

        self.embeddings.embed(sentences)
        sent = sentences[0]
        # print(sent)
        # print(sent.tokens[0].get_embedding()[0:7])

        all_sentence_tensors = []
        lengths: List[int] = []
        tag_list: List = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            # get the tags in this sentence
            tag_idx: List[int] = []

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                # get the tag
                tag_idx.append(self.tag_dictionary.get_idx_for_item(token.get_tag(tag_type)))

                word_embeddings.append(token.get_embedding().unsqueeze(0))

            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.autograd.Variable(
                        torch.FloatTensor(np.zeros(self.embeddings.embedding_length, dtype='float')).unsqueeze(0)))

            word_embeddings_tensor = torch.cat(word_embeddings, 0)

            sentence_states = word_embeddings_tensor

            if torch.cuda.is_available():
                tag_list.append(torch.cuda.LongTensor(tag_idx))
            else:
                tag_list.append(torch.LongTensor(tag_idx))

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        if torch.cuda.is_available():
            sentence_tensor = sentence_tensor.cuda()

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        tagger_states = self.dropout(sentence_tensor)

        if self.relearn_embeddings:
            tagger_states = self.embedding2nn(tagger_states)

        if self.use_RNN:
            packed = torch.nn.utils.rnn.pack_padded_sequence(tagger_states, lengths)

            rnn_output, hidden = self.rnn(packed)

            tagger_states, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

            tagger_states = self.dropout(tagger_states)

        features = self.linear(tagger_states)

        predictions_list = []
        for sentence_no, length in enumerate(lengths):
            sentence_predictions = []
            for token_no in range(length):
                sentence_predictions.append(features[token_no, sentence_no, :].unsqueeze(0))
            predictions_list.append(torch.cat(sentence_predictions, 0))

        return predictions_list, tag_list

    def _score_sentence(self, feats, tags):
        # print(tags)
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))

        if torch.cuda.is_available():
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_dictionary.get_idx_for_item(START_TAG)]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_dictionary.get_idx_for_item(STOP_TAG)])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_dictionary.get_idx_for_item(START_TAG)]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_dictionary.get_idx_for_item(STOP_TAG)])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = autograd.Variable(init_vvars)
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = autograd.Variable(torch.FloatTensor(viterbivars_t))
            if torch.cuda.is_available():
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        terminal_var.data[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.
        terminal_var.data[self.tag_dictionary.get_idx_for_item(START_TAG)] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentences: List[Sentence], tag_type: str):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        # for sentence in sentences:
        #     print(sentence)
        feats, tags = self.forward(sentences, tag_type)

        if self.use_crf:

            score = 0

            for i in range(len(feats)):
                sentence_feats = feats[i]
                sentence_tags = tags[i]

                forward_score = self._forward_alg(sentence_feats)
                # calculate the score of the ground_truth, in CRF
                gold_score = self._score_sentence(sentence_feats, sentence_tags)

                sentence_score = forward_score - gold_score
                score += sentence_score

            return score

        else:

            score = 0

            for i in range(len(feats)):
                sentence_feats = feats[i]
                sentence_tags = tags[i]

                if torch.cuda.is_available():
                    tag_tensor = autograd.Variable(torch.cuda.LongTensor(sentence_tags))
                else:
                    tag_tensor = autograd.Variable(torch.LongTensor(sentence_tags))
                score += nn.functional.cross_entropy(sentence_feats, tag_tensor)

            return score

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.
        forward_var = autograd.Variable(init_alphas)
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def predict_scores(self, sentence: Sentence, tag_type: str):
        feats, tags = self.forward([sentence], tag_type)
        feats = feats[0]
        tags = tags[0]
        # viterbi to get tag_seq
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

    def predict(self, sentence: Sentence, tag_type: str = 'tag') -> Sentence:

        score, tag_seq = self.predict_scores(sentence, tag_type)
        # sentences_out = copy.deepcopy(sentence)
        predicted_id = tag_seq
        for (token, pred_id) in zip(sentence.tokens, predicted_id):
            token: Token = token
            # get the predicted tag
            predicted_tag = self.tag_dictionary.get_item_for_index(pred_id)
            token.add_tag(tag_type, predicted_tag)

        return sentence


class LockedDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x


class Fokus(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Fokus, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        states = len(x.data[0, 0, :])
        # print(states)

        import random
        mu, sigma = 0.5, 0.2  # mean and standard deviation
        s = sorted(np.random.normal(mu, sigma, states))
        # print(s)
        mask = [0 if i < random.random() else 1 for i in s]

        if torch.cuda.is_available():
            mask = torch.autograd.Variable(torch.cuda.FloatTensor(mask), requires_grad=False)
        else:
            mask = torch.autograd.Variable(torch.FloatTensor(mask), requires_grad=False)
        # print(mask)
        # print(mask * x)

        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        # # print(x.data.new(1, x.size(1), x.size(2)))
        # # print(m)
        # mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        # mask = mask.expand_as(x)
        # asd
        return mask * x
