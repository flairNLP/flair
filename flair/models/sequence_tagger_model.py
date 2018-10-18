import warnings
import logging

import torch.autograd as autograd
import torch.nn
import flair.nn
import torch

import flair.embeddings
from flair.data import Dictionary, Sentence, Token
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import clear_embeddings

log = logging.getLogger(__name__)


START_TAG: str = '<START>'
STOP_TAG: str = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list, type_=torch.FloatTensor):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = type_(*shape)
    template.fill_(0)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, :lens_[i]] = tensor

    return template, lens_


class SequenceTagger(torch.nn.Module):

    def __init__(self,
                 hidden_size: int,
                 embeddings: flair.embeddings.TokenEmbeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 use_crf: bool = True,
                 use_rnn: bool = True,
                 rnn_layers: int = 1,
                 use_word_dropout: bool = False,
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

        # dropouts
        self.dropout: torch.nn.Module = flair.nn.LockedDropout(0.5)

        self.use_word_dropout: bool = use_word_dropout
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(0.05)

        rnn_input_dim: int = self.embeddings.embedding_length

        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        # bidirectional LSTM on top of embedding layer
        self.rnn_type = 'LSTM'
        if self.rnn_type in ['LSTM', 'GRU']:

            if self.nlayers == 1:
                self.rnn = getattr(torch.nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                            num_layers=self.nlayers,
                                                            bidirectional=True)
            else:
                self.rnn = getattr(torch.nn, self.rnn_type)(rnn_input_dim, hidden_size,
                                                            num_layers=self.nlayers,
                                                            dropout=0.5,
                                                            bidirectional=True)

        # final linear map to tag space
        if self.use_rnn:
            self.linear = torch.nn.Linear(hidden_size * 2, len(tag_dictionary))
        else:
            self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.data[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            self.transitions.data[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

        if torch.cuda.is_available():
            self.cuda()

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

    def forward(self, sentences: List[Sentence]):
        self.zero_grad()

        # first, sort sentences by number of tokens
        sentences.sort(key=lambda x: len(x), reverse=True)
        longest_token_sequence_in_batch: int = len(sentences[0])

        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros([len(sentences),
                                       longest_token_sequence_in_batch,
                                       self.embeddings.embedding_length],
                                       dtype=torch.float)

        for s_id, sentence in enumerate(sentences):

            # fill values with word embeddings
            sentence_tensor[s_id][:len(sentence)] = torch.cat([token.get_embedding().unsqueeze(0)
                                               for token in sentence], 0)

            # get the tags in this sentence
            tag_idx: List[int] = [self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                                  for token in sentence]
            # add tags as tensor
            if torch.cuda.is_available():
                tag_list.append(torch.cuda.LongTensor(tag_idx))
            else:
                tag_list.append(torch.LongTensor(tag_idx))

        sentence_tensor = sentence_tensor.transpose_(0, 1)
        if torch.cuda.is_available():
            sentence_tensor = sentence_tensor.cuda()

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        sentence_tensor = self.dropout(sentence_tensor)

        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

            rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

            sentence_tensor = self.dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features.transpose_(0, 1), lengths, tag_list

    def _score_sentence(self, feats, tags, lens_):

        if torch.cuda.is_available():
            start = torch.cuda.LongTensor([
                self.tag_dictionary.get_idx_for_item(START_TAG)
            ])
            start = start[None, :].repeat(tags.shape[0], 1)

            stop = torch.cuda.LongTensor([
                self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ])
            stop = stop[None, :].repeat(tags.shape[0], 1)

            pad_start_tags = \
                torch.cat([start, tags], 1)
            pad_stop_tags = \
                torch.cat([tags, stop], 1)
        else:
            start = torch.LongTensor([
                self.tag_dictionary.get_idx_for_item(START_TAG)
            ])
            start = start[None, :].repeat(tags.shape[0], 1)

            stop = torch.LongTensor([
                self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ])

            stop = stop[None, :].repeat(tags.shape[0], 1)

            pad_start_tags = torch.cat([start, tags], 1)
            pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = \
                self.tag_dictionary.get_idx_for_item(STOP_TAG)

        score = torch.FloatTensor(feats.shape[0])
        if torch.cuda.is_available():
            score = score.cuda()

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i]))
            if torch.cuda.is_available():
                r = r.cuda()

            score[i] = \
                torch.sum(
                    self.transitions[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]]
                ) + \
                torch.sum(feats[i, r, tags[i, :lens_[i]]])

        return score

    def viterbi_decode(self, feats):
        backpointers = []
        backscores = []

        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = autograd.Variable(init_vvars)
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        import torch.nn.functional as F
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
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        terminal_var.data[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.
        terminal_var.data[self.tag_dictionary.get_idx_for_item(START_TAG)] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())

        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()
        return best_scores, best_path

    def neg_log_likelihood(self, sentences: List[Sentence]):
        features, lengths, tags = self.forward(sentences)

        if self.use_crf:

            # pad tags if using batch-CRF decoder
            if torch.cuda.is_available():
                tags, _ = pad_tensors(tags, torch.cuda.LongTensor)
            else:
                tags, _ = pad_tensors(tags, torch.LongTensor)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.sum()

        else:

            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(features, tags, lengths):
                sentence_feats = sentence_feats[:sentence_length]

                if torch.cuda.is_available():
                    tag_tensor = autograd.Variable(torch.cuda.LongTensor(sentence_tags))
                else:
                    tag_tensor = autograd.Variable(torch.LongTensor(sentence_tags))
                score += torch.nn.functional.cross_entropy(sentence_feats, tag_tensor)

            return score

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.Tensor(self.tagset_size).fill_(-10000.)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.

        forward_var = torch.FloatTensor(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
        ).fill_(0)

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        transitions = self.transitions.view(
            1,
            self.transitions.shape[0],
            self.transitions.shape[1],
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = \
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) + \
                transitions + \
                forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1)

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - \
                      max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + \
                       self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)][None, :].repeat(
                           forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:

        if type(sentences) is Sentence:
            sentences = [sentences]

        filtered_sentences = self._filter_empty_sentences(sentences)

        # remove previous embeddings
        clear_embeddings(filtered_sentences)

        # make mini-batches
        batches = [filtered_sentences[x:x + mini_batch_size] for x in range(0, len(filtered_sentences), mini_batch_size)]

        for batch in batches:
            scores, predicted_ids = self._predict_scores_batch(batch)
            all_tokens = []
            for sentence in batch:
                all_tokens.extend(sentence.tokens)

            for (token, score, predicted_id) in zip(all_tokens, scores, predicted_ids):
                token: Token = token
                # get the predicted tag
                predicted_tag = self.tag_dictionary.get_item_for_index(predicted_id)
                token.add_tag(self.tag_type, predicted_tag, score)

        return sentences

    def _predict_scores_batch(self, sentences: List[Sentence]):
        feature, lengths, tags = self.forward(sentences)

        all_confidences = []
        all_tags_seqs = []

        for feats, length in zip(feature, lengths):

            if self.use_crf:
                confidences, tag_seq = self.viterbi_decode(feats[:length])
            else:
                import torch.nn.functional as F

                tag_seq = []
                confidences = []
                for backscore in feats[:length]:
                    softmax = F.softmax(backscore, dim=0)
                    _, idx = torch.max(backscore, 0)
                    prediction = idx.item()
                    tag_seq.append(prediction)
                    confidences.append(softmax[prediction].item())

            all_tags_seqs.extend(tag_seq)
            all_confidences.extend(confidences)

        return all_confidences, all_tags_seqs

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning('Ignore {} sentence(s) with no tokens.'.format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    @staticmethod
    def load(model: str):
        model_file = None
        aws_resource_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.2'

        if model.lower() == 'ner':
            base_path = '/'.join([aws_resource_path,
                                  'NER-conll03--h256-l1-b32-%2Bglove%2Bnews-forward%2Bnews-backward--v0.2',
                                  'en-ner-conll03-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'ner-fast':
            base_path = '/'.join([aws_resource_path,
                                  'NER-conll03--h256-l1-b32-experimental--fast-v0.2',
                                  'en-ner-fast-conll03-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'ner-ontonotes':
            base_path = '/'.join([aws_resource_path,
                                  'NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward%2Bnews-backward--v0.2',
                                  'en-ner-ontonotes-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'ner-ontonotes-fast':
            base_path = '/'.join([aws_resource_path,
                                  'NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                  'en-ner-ontonotes-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'pos':
            base_path = '/'.join([aws_resource_path,
                                  'POS-ontonotes--h256-l1-b32-%2Bmix-forward%2Bmix-backward--v0.2',
                                  'en-pos-ontonotes-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'pos-fast':
            base_path = '/'.join([aws_resource_path,
                                  'POS-ontonotes--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                  'en-pos-ontonotes-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'frame':
            base_path = '/'.join([aws_resource_path,
                                  'FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward%2Bnews-backward--v0.2',
                                  'en-frame-ontonotes-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'frame-fast':
            base_path = '/'.join([aws_resource_path,
                                  'FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                  'en-frame-ontonotes-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'chunk':
            base_path = '/'.join([aws_resource_path,
                                  'NP-conll2000--h256-l1-b32-%2Bnews-forward%2Bnews-backward--v0.2',
                                  'en-chunk-conll2000-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'chunk-fast':
            base_path = '/'.join([aws_resource_path,
                                  'NP-conll2000--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                  'en-chunk-conll2000-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-pos':
            base_path = '/'.join([aws_resource_path,
                                  'UPOS-udgerman--h256-l1-b8-%2Bgerman-forward%2Bgerman-backward--v0.2',
                                  'de-pos-ud-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-ner':
            base_path = '/'.join([aws_resource_path,
                                  'NER-conll03ger--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--v0.2',
                                  'de-ner-conll03-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model.lower() == 'de-ner-germeval':
            base_path = '/'.join([aws_resource_path,
                                  'NER-germeval--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--v0.2',
                                  'de-ner-germeval-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir='models')

        if model_file is not None:
            tagger: SequenceTagger = SequenceTagger.load_from_file(model_file)
            return tagger
