import warnings
import logging
from pathlib import Path

import torch.nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import flair.nn
import torch

import flair.embeddings
from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import clear_embeddings, Metric, Result

from tqdm import tqdm
from tabulate import tabulate

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


class SequenceTagger(flair.nn.Model):
    def __init__(
        self,
        hidden_size: int,
        embeddings: flair.embeddings.TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        pickle_module: str = "pickle",
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
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        rnn_input_dim: int = self.embeddings.embedding_length

        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        # bidirectional LSTM on top of embedding layer
        self.rnn_type = "LSTM"
        if self.rnn_type in ["LSTM", "GRU"]:

            if self.nlayers == 1:
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    bidirectional=True,
                )
            else:
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.5,
                    bidirectional=True,
                )

        # final linear map to tag space
        if self.use_rnn:
            self.linear = torch.nn.Linear(hidden_size * 2, len(tag_dictionary))
        else:
            self.linear = torch.nn.Linear(
                self.embeddings.embedding_length, len(tag_dictionary)
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
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
        }
        return model_state

    def _init_model_with_state_dict(state):

        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )

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
        )
        model.load_state_dict(state["state_dict"])
        return model

    def evaluate(
        self,
        sentences: Dataset,
        eval_mini_batch_size: int = 32,
        embeddings_in_memory: bool = True,
        out_path: Path = None,
        num_workers: int = 8,
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            batch_loader = DataLoader(
                sentences,
                batch_size=eval_mini_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            metric = Metric("Evaluation")

            lines: List[str] = []
            for batch in batch_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(features, batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                clear_embeddings(
                    batch, also_clear_word_embeddings=not embeddings_in_memory
                )

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss

    def forward_loss(
        self, sentences: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(sentences)
        return self._calculate_loss(features, sentences)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        verbose=False,
        clear_word_embeddings=True,
    ) -> List[Sentence]:
        with torch.no_grad():
            if isinstance(sentences, Sentence):
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            # remove previous embeddings
            clear_embeddings(filtered_sentences, also_clear_word_embeddings=True)

            # revere sort all sequences by their length
            filtered_sentences.sort(key=lambda x: len(x), reverse=True)

            # make mini-batches
            batches = [
                filtered_sentences[x : x + mini_batch_size]
                for x in range(0, len(filtered_sentences), mini_batch_size)
            ]

            # progress bar for verbosity
            if verbose:
                batches = tqdm(batches)

            for i, batch in enumerate(batches):

                if verbose:
                    batches.set_description(f"Inferencing on batch {i}")

                with torch.no_grad():
                    feature = self.forward(batch)
                    tags, all_tags = self._obtain_labels(feature, batch)

                for (sentence, sent_tags, sent_all_tags) in zip(batch, tags, all_tags):
                    for (token, tag, token_all_tags) in zip(
                        sentence.tokens, sent_tags, sent_all_tags
                    ):
                        token.add_tag_label(self.tag_type, tag)
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

                # clearing token embeddings to save memory
                clear_embeddings(
                    batch, also_clear_word_embeddings=clear_word_embeddings
                )

            return sentences

    def forward(self, sentences: List[Sentence]):
        self.zero_grad()

        self.embeddings.embed(sentences)

        sentences.sort(key=lambda x: len(x), reverse=True)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []
        longest_token_sequence_in_batch: int = lengths[0]

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )

            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.LongTensor(tag_idx).to(flair.device)
            tag_list.append(tag)

        sentence_tensor = sentence_tensor.transpose_(0, 1)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

            rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features.transpose_(0, 1)

    def _score_sentence(self, feats, tags, lens_):

        start = torch.LongTensor([self.tag_dictionary.get_idx_for_item(START_TAG)]).to(
            flair.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.LongTensor([self.tag_dictionary.get_idx_for_item(STOP_TAG)]).to(
            flair.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary.get_idx_for_item(
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
        self, scores: torch.tensor, sentences: List[Sentence]
    ) -> torch.tensor:

        sentences.sort(key=lambda x: len(x), reverse=True)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []

        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.LongTensor(tag_idx).to(flair.device)
            tag_list.append(tag)

        return self._calculate_loss_old(scores, lengths, tag_list)

    def _calculate_loss_old(self, features, lengths, tags) -> float:
        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(tags)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.mean()

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                features, tags, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]

                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags
                )
            score /= len(features)
            return score

    def _obtain_labels(
        self, feature, sentences
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists: 
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token 
           in a sentence for all sentences.
        """

        sentences.sort(key=lambda x: len(x), reverse=True)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tags = []
        all_tags = []
        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            else:
                tag_seq = []
                confidences = []
                scores = []
                for backscore in feats[:length]:
                    softmax = F.softmax(backscore, dim=0)
                    _, idx = torch.max(backscore, 0)
                    prediction = idx.item()
                    tag_seq.append(prediction)
                    confidences.append(softmax[prediction].item())
                    scores.append(softmax.tolist())

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            all_tags.append(
                [
                    [
                        Label(self.tag_dictionary.get_item_for_index(score_id), score)
                        for score_id, score in enumerate(score_dist)
                    ]
                    for score_dist in scores
                ]
            )

        return tags, all_tags

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []

        init_vvars = (
            torch.FloatTensor(1, self.tagset_size).to(flair.device).fill_(-10000.0)
        )
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
            forward_var
            + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        )
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.0
        terminal_var.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG)
        ] = -10000.0
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
            scores.append([elem.item() for elem in softmax.flatten()])
        # This has been taken from https://github.com/zalandoresearch/flair/pull/642
        swap_best_path, swap_max_score = (
            best_path[0],
            scores[-1].index(max(scores[-1])),
        )
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )

        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()
        return best_scores, best_path, scores

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
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def _fetch_model(model_name) -> str:

        model_map = {}
        aws_resource_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.2"
        )
        aws_resource_path_v04 = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
        )

        model_map["ner"] = "/".join(
            [aws_resource_path_v04, "NER-conll03-english", "en-ner-conll03-v0.4.pt"]
        )

        model_map["ner-fast"] = "/".join(
            [
                aws_resource_path,
                "NER-conll03--h256-l1-b32-experimental--fast-v0.2",
                "en-ner-fast-conll03-v0.2.pt",
            ]
        )

        model_map["ner-ontonotes"] = "/".join(
            [
                aws_resource_path,
                "NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward%2Bnews-backward--v0.2",
                "en-ner-ontonotes-v0.3.pt",
            ]
        )

        model_map["ner-ontonotes-fast"] = "/".join(
            [
                aws_resource_path,
                "NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward-fast%2Bnews-backward-fast--v0.2",
                "en-ner-ontonotes-fast-v0.3.pt",
            ]
        )

        for key in ["ner-multi", "multi-ner"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-quadner-512-l2-multi-embed",
                    "quadner-large.pt",
                ]
            )

        for key in ["ner-multi-fast", "multi-ner-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "NER-multi-fast", "ner-multi-fast.pt"]
            )

        for key in ["ner-multi-fast-learn", "multi-ner-fast-learn"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "NER-multi-fast-evolve",
                    "ner-multi-fast-learn.pt",
                ]
            )

        model_map["pos"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-ontonotes--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-pos-ontonotes-v0.4.pt",
            ]
        )

        model_map["pos-fast"] = "/".join(
            [
                aws_resource_path,
                "POS-ontonotes--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2",
                "en-pos-ontonotes-fast-v0.2.pt",
            ]
        )

        for key in ["pos-multi", "multi-pos"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-dodekapos-512-l2-multi",
                    "pos-multi-v0.1.pt",
                ]
            )

        for key in ["pos-multi-fast", "multi-pos-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "UPOS-multi-fast", "pos-multi-fast.pt"]
            )

        model_map["frame"] = "/".join(
            [
                aws_resource_path,
                "FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward%2Bnews-backward--v0.2",
                "en-frame-ontonotes-v0.2.pt",
            ]
        )

        model_map["frame-fast"] = "/".join(
            [
                aws_resource_path,
                "FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward-fast%2Bnews-backward-fast--v0.2",
                "en-frame-ontonotes-fast-v0.2.pt",
            ]
        )

        model_map["chunk"] = "/".join(
            [
                aws_resource_path_v04,
                "NP-conll2000--h256-l1-b32-p3-0.5-%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-chunk-conll2000-v0.4.pt",
            ]
        )

        model_map["chunk-fast"] = "/".join(
            [
                aws_resource_path,
                "NP-conll2000--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2",
                "en-chunk-conll2000-fast-v0.2.pt",
            ]
        )

        model_map["de-pos"] = "/".join(
            [
                aws_resource_path,
                "UPOS-udgerman--h256-l1-b8-%2Bgerman-forward%2Bgerman-backward--v0.2",
                "de-pos-ud-v0.2.pt",
            ]
        )

        model_map["de-pos-fine-grained"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-fine-grained-german-tweets",
                "de-pos-twitter-v0.1.pt",
            ]
        )

        model_map["de-ner"] = "/".join(
            [
                aws_resource_path,
                "NER-conll03ger--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--v0.2",
                "de-ner-conll03-v0.3.pt",
            ]
        )

        model_map["de-ner-germeval"] = "/".join(
            [aws_resource_path_v04, "NER-germeval", "de-ner-germeval-0.4.1.pt"]
        )

        model_map["fr-ner"] = "/".join(
            [aws_resource_path, "NER-aij-wikiner-fr-wp3", "fr-ner.pt"]
        )
        model_map["nl-ner"] = "/".join(
            [aws_resource_path_v04, "NER-conll2002-dutch", "nl-ner-conll02-v0.1.pt"]
        )

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

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
