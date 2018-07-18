import warnings

import torch.autograd as autograd
import torch.nn as nn
import torch
import os
import numpy as np

from flair.file_utils import cached_path
from .data import Dictionary, Sentence, Token
from .embeddings import TextEmbeddings

from typing import List, Tuple, Union

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


class SequenceTagger(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 embeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 use_crf: bool = True,
                 use_rnn: bool = True,
                 rnn_layers: int = 1
                 ):

        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
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
        if self.use_rnn:
            self.linear = nn.Linear(hidden_size * 2, len(tag_dictionary))
        else:
            self.linear = nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        # trans is also a score tensor, not a probability: THIS THING NEEDS TO GO!!!!
        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.data[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            self.transitions.data[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

    def save(self, model_file: str):
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'tag_dictionary': self.tag_dictionary,
            'tag_type': self.tag_type,
            'use_crf': self.use_crf,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
        }
        torch.save(model_state, model_file, pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file):

        # ACHTUNG: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        warnings.filterwarnings("ignore")
        state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
        warnings.filterwarnings("default")

        model = SequenceTagger(
            hidden_size=state['hidden_size'],
            embeddings=state['embeddings'],
            tag_dictionary=state['tag_dictionary'],
            tag_type=state['tag_type'],
            use_crf=state['use_crf'],
            use_rnn=state['use_rnn'],
            rnn_layers=state['rnn_layers'])

        model.load_state_dict(state['state_dict'])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def forward(self, sentences: List[Sentence]) -> Tuple[List, List]:

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
                tag_idx.append(self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type)))

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

        if self.use_rnn:
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
        feats, tags = self.forward(sentences)

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

    def predict_scores(self, sentence: Sentence):
        feats, tags = self.forward([sentence])
        feats = feats[0]
        tags = tags[0]
        # viterbi to get tag_seq
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

    def predict_old(self, sentence: Sentence) -> Sentence:

        score, tag_seq = self.predict_scores(sentence)
        predicted_id = tag_seq
        for (token, pred_id) in zip(sentence.tokens, predicted_id):
            token: Token = token
            # get the predicted tag
            predicted_tag = self.tag_dictionary.get_item_for_index(pred_id)
            token.add_tag(self.tag_type, predicted_tag)

        return sentence

    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:

        if type(sentences) is Sentence:
            sentences = [sentences]

        # make mini-batches
        batches = [sentences[x:x + mini_batch_size] for x in range(0, len(sentences), mini_batch_size)]

        for batch in batches:
            score, tag_seq = self._predict_scores_batch(batch)
            predicted_id = tag_seq
            all_tokens = []
            for sentence in batch:
                all_tokens.extend(sentence.tokens)

            for (token, pred_id) in zip(all_tokens, predicted_id):
                token: Token = token
                # get the predicted tag
                predicted_tag = self.tag_dictionary.get_item_for_index(pred_id)
                token.add_tag(self.tag_type, predicted_tag)

        return sentences

    def _predict_scores_batch(self, sentences: List[Sentence]):
        all_feats, tags = self.forward(sentences)

        overall_score = 0
        all_tags_seqs = []

        for feats in all_feats:
            # viterbi to get tag_seq
            if self.use_crf:
                score, tag_seq = self.viterbi_decode(feats)
            else:
                score, tag_seq = torch.max(feats, 1)
                tag_seq = list(tag_seq.cpu().data)

            # overall_score += score
            all_tags_seqs.extend(tag_seq)

        return overall_score, all_tags_seqs

    @staticmethod
    def load(model: str):
        model_file = None
        aws_resource_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models'

        if model.lower() == 'ner':
            base_path = '/'.join([aws_resource_path,
                                  'NER-conll03--h256-l1-b32-%2Bglove%2Bnews-forward%2Bnews-backward--anneal',
                                  'en-ner-conll03-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'ner-ontonotes':
            base_path = '/'.join([aws_resource_path,
                                  'NER-ontoner--h256-l1-b32-%2Bft-crawl%2Bnews-forward%2Bnews-backward--anneal',
                                  'en-ner-ontonotes-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'chunk':
            base_path = '/'.join([aws_resource_path,
                                  'NP-conll2000--h256-l1-b32-%2Bnews-forward%2Bnews-backward--anneal',
                                  'en-chunk-conll2000-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'pos':
            base_path = '/'.join([aws_resource_path,
                                  'POS-ontonotes--h256-l1-b32-%2Bmix-forward%2Bmix-backward--anneal',
                                  'en-pos-ontonotes-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'frame':
            base_path = '/'.join([aws_resource_path,
                                  'FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward%2Bnews-backward--anneal',
                                  'en-frame-ontonotes-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-pos':
            base_path = '/'.join([aws_resource_path,
                                  'UPOS-udgerman--h256-l1-b8-%2Bgerman-forward%2Bgerman-backward--anneal',
                                  'de-pos-ud-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-ner':
            base_path = '/'.join([aws_resource_path,
                                  'NER-conll03ger--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--anneal',
                                  'de-ner-conll03-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-ner-germeval':
            base_path = '/'.join([aws_resource_path,
                                  'NER-germeval--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--anneal',
                                  'de-ner-germeval-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model_file is not None:
            tagger: SequenceTagger = SequenceTagger.load_from_file(model_file)
            return tagger


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
