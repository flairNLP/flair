import logging
import sys

from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from requests import HTTPError
from tabulate import tabulate
from torch.nn.parameter import Parameter
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, Embeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import store_embeddings

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template, lens_


class SequenceTagger(flair.nn.Classifier):
    def __init__(
            self,
            hidden_size: int,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_crf: bool = True,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            dropout: float = 0.0,
            word_dropout: float = 0.05,
            locked_dropout: float = 0.5,
            reproject_embeddings: Union[bool, int] = True,
            train_initial_hidden_state: bool = False,
            rnn_type: str = "LSTM",
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        If you set this to an integer, you can control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """
        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        # if we use a CRF, we must add special START and STOP tags to the dictionary
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        embedding_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = embedding_dim

        # optional reprojection layer on top of word embeddings
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim = self.reproject_embeddings

            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = rnn_type

        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    # TODO: Decide how to initialize the hidden state variables
                    # self.hs_initializer(self.lstm_init_h)
                    # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, len(tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                rnn_input_dim, len(tag_dictionary)
            )

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )

            self.transitions.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000

            self.transitions.detach()[
            :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "reproject_embeddings": self.reproject_embeddings,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = 0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        use_locked_dropout = 0.0 if "use_locked_dropout" not in state.keys() else state["use_locked_dropout"]

        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]

        model = SequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence):
                sentences = [sentences]

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            overall_count = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature = self.forward(batch)

                if return_loss:
                    loss_and_count = self._calculate_loss(feature, batch)
                    overall_loss += loss_and_count[0]
                    overall_count += loss_and_count[1]

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, overall_count

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    def forward(self, sentences: List[Sentence]):

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features

    def _score_sentence(self, feats, tags, lens_):

        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = self.tag_dictionary.get_idx_for_item(
                STOP_TAG
            )

        score = torch.FloatTensor(feats.shape[0]).to(flair.device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(flair.device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score

    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> Tuple[float, int]:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        token_count = 0
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            token_count += len(tag_idx)
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(tag_list)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.sum(), token_count

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                    features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags, weight=self.loss_weights, reduction='sum',
                )

            return score, token_count

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            transitions: Optional[np.ndarray],
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        feature = feature.cpu()
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
                softmax, score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
                scores = softmax[:length].tolist()

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                self.tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

        return tags, all_tags

    @staticmethod
    def _softmax(x, axis):
        # reduce raw values to avoid NaN during exp
        x_norm = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x_norm)
        return y / y.sum(axis=axis, keepdims=True)

    def _viterbi_decode(
            self, feats: np.ndarray, transitions: np.ndarray, all_scores: bool
    ):
        id_start = self.tag_dictionary.get_idx_for_item(START_TAG)
        id_stop = self.tag_dictionary.get_idx_for_item(STOP_TAG)

        backpointers = np.empty(shape=(feats.shape[0], self.tagset_size), dtype=np.int_)
        backscores = np.empty(
            shape=(feats.shape[0], self.tagset_size), dtype=np.float32
        )

        init_vvars = np.expand_dims(
            np.repeat(-10000.0, self.tagset_size), axis=0
        ).astype(np.float32)
        init_vvars[0][id_start] = 0

        forward_var = init_vvars
        for index, feat in enumerate(feats):
            # broadcasting will do the job of reshaping and is more efficient than calling repeat
            next_tag_var = forward_var + transitions
            bptrs_t = next_tag_var.argmax(axis=1)
            viterbivars_t = next_tag_var[np.arange(bptrs_t.shape[0]), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores[index] = forward_var
            forward_var = forward_var[np.newaxis, :]
            backpointers[index] = bptrs_t

        terminal_var = forward_var.squeeze() + transitions[id_stop]
        terminal_var[id_stop] = -10000.0
        terminal_var[id_start] = -10000.0
        best_tag_id = terminal_var.argmax()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == id_start
        best_path.reverse()

        best_scores_softmax = self._softmax(backscores, axis=1)
        best_scores_np = np.max(best_scores_softmax, axis=1)

        # default value
        all_scores_np = np.zeros(0, dtype=np.float64)
        if all_scores:
            all_scores_np = best_scores_softmax
            for index, (tag_id, tag_scores) in enumerate(zip(best_path, all_scores_np)):
                if type(tag_id) != int and tag_id.item() != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    (
                        all_scores_np[index][tag_id.item()],
                        all_scores_np[index][swap_index_score],
                    ) = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id.item()],
                    )
                elif type(tag_id) == int and tag_id != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    (
                        all_scores_np[index][tag_id],
                        all_scores_np[index][swap_index_score],
                    ) = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id],
                    )

        return best_scores_np.tolist(), best_path, all_scores_np.tolist()

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=flair.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
                                         self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _filter_empty_string(texts: List[str]) -> List[str]:
        filtered_texts = [text for text in texts if text]
        if len(texts) != len(filtered_texts):
            log.warning(
                f"Ignore {len(texts) - len(filtered_texts)} string(s) with no tokens."
            )
        return filtered_texts

    @staticmethod
    def _fetch_model(model_name) -> str:

        # core Flair models on Huggingface ModelHub
        huggingface_model_map = {
            "ner": "flair/ner-english",
            "ner-fast": "flair/ner-english-fast",
            "ner-ontonotes": "flair/ner-english-ontonotes",
            "ner-ontonotes-fast": "flair/ner-english-ontonotes-fast",
            # Large NER models,
            "ner-large": "flair/ner-english-large",
            "ner-ontonotes-large": "flair/ner-english-ontonotes-large",
            "de-ner-large": "flair/ner-german-large",
            "nl-ner-large": "flair/ner-dutch-large",
            "es-ner-large": "flair/ner-spanish-large",
            # Multilingual NER models
            "ner-multi": "flair/ner-multi",
            "multi-ner": "flair/ner-multi",
            "ner-multi-fast": "flair/ner-multi-fast",
            # English POS models
            "upos": "flair/upos-english",
            "upos-fast": "flair/upos-english-fast",
            "pos": "flair/pos-english",
            "pos-fast": "flair/pos-english-fast",
            # Multilingual POS models
            "pos-multi": "flair/upos-multi",
            "multi-pos": "flair/upos-multi",
            "pos-multi-fast": "flair/upos-multi-fast",
            "multi-pos-fast": "flair/upos-multi-fast",
            # English SRL models
            "frame": "flair/frame-english",
            "frame-fast": "flair/frame-english-fast",
            # English chunking models
            "chunk": "flair/chunk-english",
            "chunk-fast": "flair/chunk-english-fast",
            # Language-specific NER models
            "ar-ner": "megantosh/flair-arabic-multi-ner",
            "ar-pos": "megantosh/flair-arabic-dialects-codeswitch-egy-lev",
            "da-ner": "flair/ner-danish",
            "de-ner": "flair/ner-german",
            "de-ler": "flair/ner-german-legal",
            "de-ner-legal": "flair/ner-german-legal",
            "fr-ner": "flair/ner-french",
            "nl-ner": "flair/ner-dutch",
        }

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        hu_model_map = {
            # English NER models
            "ner": "/".join([hu_path, "ner", "en-ner-conll03-v0.4.pt"]),
            "ner-pooled": "/".join([hu_path, "ner-pooled", "en-ner-conll03-pooled-v0.5.pt"]),
            "ner-fast": "/".join([hu_path, "ner-fast", "en-ner-fast-conll03-v0.4.pt"]),
            "ner-ontonotes": "/".join([hu_path, "ner-ontonotes", "en-ner-ontonotes-v0.4.pt"]),
            "ner-ontonotes-fast": "/".join([hu_path, "ner-ontonotes-fast", "en-ner-ontonotes-fast-v0.4.pt"]),
            # Multilingual NER models
            "ner-multi": "/".join([hu_path, "multi-ner", "quadner-large.pt"]),
            "multi-ner": "/".join([hu_path, "multi-ner", "quadner-large.pt"]),
            "ner-multi-fast": "/".join([hu_path, "multi-ner-fast", "ner-multi-fast.pt"]),
            # English POS models
            "upos": "/".join([hu_path, "upos", "en-pos-ontonotes-v0.4.pt"]),
            "upos-fast": "/".join([hu_path, "upos-fast", "en-upos-ontonotes-fast-v0.4.pt"]),
            "pos": "/".join([hu_path, "pos", "en-pos-ontonotes-v0.5.pt"]),
            "pos-fast": "/".join([hu_path, "pos-fast", "en-pos-ontonotes-fast-v0.5.pt"]),
            # Multilingual POS models
            "pos-multi": "/".join([hu_path, "multi-pos", "pos-multi-v0.1.pt"]),
            "multi-pos": "/".join([hu_path, "multi-pos", "pos-multi-v0.1.pt"]),
            "pos-multi-fast": "/".join([hu_path, "multi-pos-fast", "pos-multi-fast.pt"]),
            "multi-pos-fast": "/".join([hu_path, "multi-pos-fast", "pos-multi-fast.pt"]),
            # English SRL models
            "frame": "/".join([hu_path, "frame", "en-frame-ontonotes-v0.4.pt"]),
            "frame-fast": "/".join([hu_path, "frame-fast", "en-frame-ontonotes-fast-v0.4.pt"]),
            # English chunking models
            "chunk": "/".join([hu_path, "chunk", "en-chunk-conll2000-v0.4.pt"]),
            "chunk-fast": "/".join([hu_path, "chunk-fast", "en-chunk-conll2000-fast-v0.4.pt"]),
            # Danish models
            "da-pos": "/".join([hu_path, "da-pos", "da-pos-v0.1.pt"]),
            "da-ner": "/".join([hu_path, "NER-danish", "da-ner-v0.1.pt"]),
            # German models
            "de-pos": "/".join([hu_path, "de-pos", "de-pos-ud-hdt-v0.5.pt"]),
            "de-pos-tweets": "/".join([hu_path, "de-pos-tweets", "de-pos-twitter-v0.1.pt"]),
            "de-ner": "/".join([hu_path, "de-ner", "de-ner-conll03-v0.4.pt"]),
            "de-ner-germeval": "/".join([hu_path, "de-ner-germeval", "de-ner-germeval-0.4.1.pt"]),
            "de-ler": "/".join([hu_path, "de-ner-legal", "de-ner-legal.pt"]),
            "de-ner-legal": "/".join([hu_path, "de-ner-legal", "de-ner-legal.pt"]),
            # French models
            "fr-ner": "/".join([hu_path, "fr-ner", "fr-ner-wikiner-0.4.pt"]),
            # Dutch models
            "nl-ner": "/".join([hu_path, "nl-ner", "nl-ner-bert-conll02-v0.8.pt"]),
            "nl-ner-rnn": "/".join([hu_path, "nl-ner-rnn", "nl-ner-conll02-v0.5.pt"]),
            # Malayalam models
            "ml-pos": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-xpos-model.pt",
            "ml-upos": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-upos-model.pt",
            # Portuguese models
            "pt-pos-clinical": "/".join([hu_path, "pt-pos-clinical", "pucpr-flair-clinical-pos-tagging-best-model.pt"]),
            # Keyphase models
            "keyphrase": "/".join([hu_path, "keyphrase", "keyphrase-en-scibert.pt"]),
            "negation-speculation": "/".join(
                [hu_path, "negation-speculation", "negation-speculation-model.pt"]),
            # Biomedical models
            "hunflair-paper-cellline": "/".join(
                [hu_path, "hunflair_smallish_models", "cellline", "hunflair-celline-v1.0.pt"]
            ),
            "hunflair-paper-chemical": "/".join(
                [hu_path, "hunflair_smallish_models", "chemical", "hunflair-chemical-v1.0.pt"]
            ),
            "hunflair-paper-disease": "/".join(
                [hu_path, "hunflair_smallish_models", "disease", "hunflair-disease-v1.0.pt"]
            ),
            "hunflair-paper-gene": "/".join(
                [hu_path, "hunflair_smallish_models", "gene", "hunflair-gene-v1.0.pt"]
            ),
            "hunflair-paper-species": "/".join(
                [hu_path, "hunflair_smallish_models", "species", "hunflair-species-v1.0.pt"]
            ),
            "hunflair-cellline": "/".join(
                [hu_path, "hunflair_smallish_models", "cellline", "hunflair-celline-v1.0.pt"]
            ),
            "hunflair-chemical": "/".join(
                [hu_path, "hunflair_allcorpus_models", "huner-chemical", "hunflair-chemical-full-v1.0.pt"]
            ),
            "hunflair-disease": "/".join(
                [hu_path, "hunflair_allcorpus_models", "huner-disease", "hunflair-disease-full-v1.0.pt"]
            ),
            "hunflair-gene": "/".join(
                [hu_path, "hunflair_allcorpus_models", "huner-gene", "hunflair-gene-full-v1.0.pt"]
            ),
            "hunflair-species": "/".join(
                [hu_path, "hunflair_allcorpus_models", "huner-species", "hunflair-species-full-v1.1.pt"]
            )}

        cache_dir = Path("models")

        get_from_model_hub = False

        # check if model name is a valid local file
        if Path(model_name).exists():
            model_path = model_name

        # check if model key is remapped to HF key - if so, print out information
        elif model_name in huggingface_model_map:

            # get mapped name
            hf_model_name = huggingface_model_map[model_name]

            # output information
            log.info("-" * 80)
            log.info(
                f"The model key '{model_name}' now maps to 'https://huggingface.co/{hf_model_name}' on the HuggingFace ModelHub")
            log.info(f" - The most current version of the model is automatically downloaded from there.")
            if model_name in hu_model_map:
                log.info(
                    f" - (you can alternatively manually download the original model at {hu_model_map[model_name]})")
            log.info("-" * 80)

            # use mapped name instead
            model_name = hf_model_name
            get_from_model_hub = True

        # if not, check if model key is remapped to direct download location. If so, download model
        elif model_name in hu_model_map:
            model_path = cached_path(hu_model_map[model_name], cache_dir=cache_dir)

        # special handling for the taggers by the @redewiegergabe project (TODO: move to model hub)
        elif model_name == "de-historic-indirect":
            model_file = flair.cache_root / cache_dir / 'indirect' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/indirect.zip', cache_dir=cache_dir)
                unzip_file(flair.cache_root / cache_dir / 'indirect.zip', flair.cache_root / cache_dir)
            model_path = str(flair.cache_root / cache_dir / 'indirect' / 'final-model.pt')

        elif model_name == "de-historic-direct":
            model_file = flair.cache_root / cache_dir / 'direct' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/direct.zip', cache_dir=cache_dir)
                unzip_file(flair.cache_root / cache_dir / 'direct.zip', flair.cache_root / cache_dir)
            model_path = str(flair.cache_root / cache_dir / 'direct' / 'final-model.pt')

        elif model_name == "de-historic-reported":
            model_file = flair.cache_root / cache_dir / 'reported' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/reported.zip', cache_dir=cache_dir)
                unzip_file(flair.cache_root / cache_dir / 'reported.zip', flair.cache_root / cache_dir)
            model_path = str(flair.cache_root / cache_dir / 'reported' / 'final-model.pt')

        elif model_name == "de-historic-free-indirect":
            model_file = flair.cache_root / cache_dir / 'freeIndirect' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/freeIndirect.zip', cache_dir=cache_dir)
                unzip_file(flair.cache_root / cache_dir / 'freeIndirect.zip', flair.cache_root / cache_dir)
            model_path = str(flair.cache_root / cache_dir / 'freeIndirect' / 'final-model.pt')

        # for all other cases (not local file or special download location), use HF model hub
        else:
            get_from_model_hub = True

        # if not a local file, get from model hub
        if get_from_model_hub:
            hf_model_name = "pytorch_model.bin"
            revision = "main"

            if "@" in model_name:
                model_name_split = model_name.split("@")
                revision = model_name_split[-1]
                model_name = model_name_split[0]

            # use model name as subfolder
            if "/" in model_name:
                model_folder = model_name.split("/", maxsplit=1)[1]
            else:
                model_folder = model_name

            # Lazy import
            from huggingface_hub import hf_hub_url, cached_download

            url = hf_hub_url(model_name, revision=revision, filename=hf_model_name)

            try:
                model_path = cached_download(url=url, library_name="flair",
                                             library_version=flair.__version__,
                                             cache_dir=flair.cache_root / 'models' / model_folder)
            except HTTPError as e:
                # output information
                log.error("-" * 80)
                log.error(
                    f"ACHTUNG: The key '{model_name}' was neither found on the ModelHub nor is this a valid path to a file on your system!")
                # log.error(f" - Error message: {e}")
                log.error(f" -> Please check https://huggingface.co/models?filter=flair for all available models.")
                log.error(f" -> Alternatively, point to a model file on your local drive.")
                log.error("-" * 80)
                Path(flair.cache_root / 'models' / model_folder).rmdir()  # remove folder again if not valid

        return model_path

    def get_transition_matrix(self):
        data = []
        for to_idx, row in enumerate(self.transitions):
            for from_idx, column in enumerate(row):
                row = [
                    self.tag_dictionary.get_item_for_index(from_idx),
                    self.tag_dictionary.get_item_for_index(to_idx),
                    column.item(),
                ]
                data.append(row)
            data.append(["----"])
        print(tabulate(data, headers=["FROM", "TO", "SCORE"]))

    def __str__(self):
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)'

    @property
    def label_type(self):
        return self.tag_type


class MultiTagger:
    def __init__(self, name_to_tagger: Dict[str, SequenceTagger]):
        super().__init__()
        self.name_to_tagger = name_to_tagger

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            return_loss: bool = False,
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        """
        if any(["hunflair" in name for name in self.name_to_tagger.keys()]):
            if "spacy" not in sys.modules:
                warn(
                    "We recommend to use SciSpaCy for tokenization and sentence splitting "
                    "if HunFlair is applied to biomedical text, e.g.\n\n"
                    "from flair.tokenization import SciSpacySentenceSplitter\n"
                    "sentence = Sentence('Your biomed text', use_tokenizer=SciSpacySentenceSplitter())\n"
                )

        if isinstance(sentences, Sentence):
            sentences = [sentences]
        for name, tagger in self.name_to_tagger.items():
            tagger.predict(
                sentences=sentences,
                mini_batch_size=mini_batch_size,
                # all_tag_prob=all_tag_prob,
                verbose=verbose,
                label_name=name,
                return_loss=return_loss,
                embedding_storage_mode="cpu",
            )

        # clear embeddings after predicting
        for sentence in sentences:
            sentence.clear_embeddings()

    @classmethod
    def load(cls, model_names: Union[List[str], str]):
        if model_names == "hunflair-paper":
            model_names = [
                "hunflair-paper-cellline",
                "hunflair-paper-chemical",
                "hunflair-paper-disease",
                "hunflair-paper-gene",
                "hunflair-paper-species",
            ]
        elif model_names == "hunflair" or model_names == "bioner":
            model_names = [
                "hunflair-cellline",
                "hunflair-chemical",
                "hunflair-disease",
                "hunflair-gene",
                "hunflair-species",
            ]
        elif isinstance(model_names, str):
            model_names = [model_names]

        taggers = {}
        models = []

        # load each model
        for model_name in model_names:

            model = SequenceTagger.load(model_name)

            # check if the same embeddings were already loaded previously
            # if the model uses StackedEmbedding, make a new stack with previous objects
            if type(model.embeddings) == StackedEmbeddings:

                # sort embeddings by key alphabetically
                new_stack = []
                d = model.embeddings.get_named_embeddings_dict()
                import collections
                od = collections.OrderedDict(sorted(d.items()))

                for k, embedding in od.items():

                    # check previous embeddings and add if found
                    embedding_found = False
                    for previous_model in models:

                        # only re-use static embeddings
                        if not embedding.static_embeddings: continue

                        if embedding.name in previous_model.embeddings.get_named_embeddings_dict():
                            previous_embedding = previous_model.embeddings.get_named_embeddings_dict()[embedding.name]
                            previous_embedding.name = previous_embedding.name[2:]
                            new_stack.append(previous_embedding)
                            embedding_found = True
                            break

                    # if not found, use existing embedding
                    if not embedding_found:
                        embedding.name = embedding.name[2:]
                        new_stack.append(embedding)

                # initialize new stack
                model.embeddings = None
                model.embeddings = StackedEmbeddings(new_stack)

            else:
                # of the model uses regular embedding, re-load if previous version found
                if not model.embeddings.static_embeddings:

                    for previous_model in models:
                        if model.embeddings.name in previous_model.embeddings.get_named_embeddings_dict():
                            previous_embedding = previous_model.embeddings.get_named_embeddings_dict()[
                                model.embeddings.name]
                            if not previous_embedding.static_embeddings:
                                model.embeddings = previous_embedding
                                break

            taggers[model_name] = model
            models.append(model)

        return cls(taggers)
