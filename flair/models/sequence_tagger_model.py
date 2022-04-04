import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Label, Sentence, Span
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import StackedEmbeddings, TokenEmbeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import store_embeddings

from .sequence_tagger_utils.bioes import get_spans_from_bio
from .sequence_tagger_utils.crf import CRF
from .sequence_tagger_utils.viterbi import ViterbiDecoder, ViterbiLoss

log = logging.getLogger("flair")


class SequenceTagger(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_rnn: bool = True,
        rnn: Optional[torch.nn.RNN] = None,
        rnn_type: str = "LSTM",
        tag_format: str = "BIOES",
        hidden_size: int = 256,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        use_crf: bool = True,
        reproject_embeddings: bool = True,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        train_initial_hidden_state: bool = False,
        loss_weights: Dict[str, float] = None,
        init_from_state_dict: bool = False,
        allow_unk_predictions: bool = False,
    ):
        """
        Sequence Tagger class for predicting labels for single tokens. Can be parameterized by several attributes.
        In case of multitask learning, pass shared embeddings or shared rnn into respective attributes.
        :param embeddings: Embeddings to use during training and prediction
        :param tag_dictionary: Dictionary containing all tags from corpus which can be predicted
        :param tag_type: type of tag which is going to be predicted in case a corpus has multiple annotations
        :param use_rnn: If true, use a RNN, else Linear layer.
        :param rnn: (Optional) Takes a torch.nn.Module as parameter by which you can pass a shared RNN between
            different tasks.
        :param rnn_type: Specifies the RNN type to use, default is 'LSTM', can choose between 'GRU' and 'RNN' as well.
        :param hidden_size: Hidden size of RNN layer
        :param rnn_layers: number of RNN layers
        :param bidirectional: If True, RNN becomes bidirectional
        :param use_crf: If True, use a Conditional Random Field for prediction, else linear map to tag space.
        :param reproject_embeddings: If True, add a linear layer on top of embeddings, if you want to imitate
            fine tune non-trainable embeddings.
        :param dropout: If > 0, then use dropout.
        :param word_dropout: If > 0, then use word dropout.
        :param locked_dropout: If > 0, then use locked dropout.
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        :param init_from_state_dict: Indicator whether we are loading a model from state dict
            since we need to transform previous models' weights into CRF instance weights
        """
        super(SequenceTagger, self).__init__()

        # ----- Create the internal tag dictionary -----
        self.tag_type = tag_type
        self.tag_format = tag_format.upper()
        if init_from_state_dict:
            self.label_dictionary = tag_dictionary
        else:
            # span-labels need special encoding (BIO or BIOES)
            if tag_dictionary.span_labels:
                # the big question is whether the label dictionary should contain an UNK or not
                # without UNK, we cannot evaluate on data that contains labels not seen in test
                # with UNK, the model learns less well if there are no UNK examples
                self.label_dictionary = Dictionary(add_unk=allow_unk_predictions)
                for label in tag_dictionary.get_items():
                    if label == "<unk>":
                        continue
                    self.label_dictionary.add_item("O")
                    if tag_format == "BIOES":
                        self.label_dictionary.add_item("S-" + label)
                        self.label_dictionary.add_item("B-" + label)
                        self.label_dictionary.add_item("E-" + label)
                        self.label_dictionary.add_item("I-" + label)
                    if tag_format == "BIO":
                        self.label_dictionary.add_item("B-" + label)
                        self.label_dictionary.add_item("I-" + label)
            else:
                self.label_dictionary = tag_dictionary

        # is this a span prediction problem?
        self.predict_spans = self._determine_if_span_prediction_problem(self.label_dictionary)

        self.tagset_size = len(self.label_dictionary)
        log.info(f"SequenceTagger predicts: {self.label_dictionary}")

        # ----- Embeddings -----
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length

        # ----- Initial loss weights parameters -----
        self.weight_dict = loss_weights
        self.loss_weights = self._init_loss_weights(loss_weights) if loss_weights else None

        # ----- RNN specific parameters -----
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type if not rnn else rnn._get_name()
        self.hidden_size = hidden_size if not rnn else rnn.hidden_size
        self.rnn_layers = rnn_layers if not rnn else rnn.num_layers
        self.bidirectional = bidirectional if not rnn else rnn.bidirectional

        # ----- Conditional Random Field parameters -----
        self.use_crf = use_crf
        # Previously trained models have been trained without an explicit CRF, thus it is required to check
        # whether we are loading a model from state dict in order to skip or add START and STOP token
        if use_crf and not init_from_state_dict and not self.label_dictionary.start_stop_tags_are_set():
            self.label_dictionary.set_start_stop_tags()
            self.tagset_size += 2

        # ----- Dropout parameters -----
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

        # ----- Model layers -----
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        # ----- RNN layer -----
        if use_rnn:
            # If shared RNN provided, else create one for model
            self.rnn: torch.nn.RNN = (
                rnn
                if rnn
                else self.RNN(
                    rnn_type,
                    rnn_layers,
                    hidden_size,
                    bidirectional,
                    rnn_input_dim=embedding_dim,
                )
            )

            num_directions = 2 if self.bidirectional else 1
            hidden_output_dim = self.rnn.hidden_size * num_directions

            # Whether to train initial hidden state
            self.train_initial_hidden_state = train_initial_hidden_state
            if self.train_initial_hidden_state:
                (
                    self.hs_initializer,
                    self.lstm_init_h,
                    self.lstm_init_c,
                ) = self._init_initial_hidden_state(num_directions)

            # final linear map to tag space
            self.linear = torch.nn.Linear(hidden_output_dim, len(self.label_dictionary))
        else:
            self.linear = torch.nn.Linear(embedding_dim, len(self.label_dictionary))

        # the loss function is Viterbi if using CRF, else regular Cross Entropy Loss
        self.loss_function = (
            ViterbiLoss(self.label_dictionary)
            if use_crf
            else torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")
        )

        # if using CRF, we also require a CRF and a Viterbi decoder
        if use_crf:
            self.crf = CRF(self.label_dictionary, self.tagset_size, init_from_state_dict)
            self.viterbi_decoder = ViterbiDecoder(self.label_dictionary)

        self.to(flair.device)

    @property
    def label_type(self):
        return self.tag_type

    def _init_loss_weights(self, loss_weights: Dict[str, float]) -> torch.Tensor:
        """
        Intializes the loss weights based on given dictionary:
        :param loss_weights: dictionary - contains loss weights
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1.0 for _ in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]

        return torch.tensor(weight_list).to(flair.device)

    def _init_initial_hidden_state(self, num_directions: int):
        """
        Intializes hidden states given the number of directions in RNN.
        :param num_directions: Number of directions in RNN.
        """
        hs_initializer = torch.nn.init.xavier_normal_
        lstm_init_h = torch.nn.Parameter(
            torch.randn(self.rnn.num_layers * num_directions, self.hidden_size),
            requires_grad=True,
        )
        lstm_init_c = torch.nn.Parameter(
            torch.randn(self.rnn.num_layers * num_directions, self.hidden_size),
            requires_grad=True,
        )

        return hs_initializer, lstm_init_h, lstm_init_c

    @staticmethod
    def RNN(
        rnn_type: str,
        rnn_layers: int,
        hidden_size: int,
        bidirectional: bool,
        rnn_input_dim: int,
    ) -> torch.nn.RNN:
        """
        Static wrapper function returning an RNN instance from PyTorch
        :param rnn_type: Type of RNN from torch.nn
        :param rnn_layers: number of layers to include
        :param hidden_size: hidden size of RNN cell
        :param bidirectional: If True, RNN cell is bidirectional
        :param rnn_input_dim: Input dimension to RNN cell
        """
        if rnn_type in ["LSTM", "GRU", "RNN"]:
            RNN = getattr(torch.nn, rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=rnn_layers,
                dropout=0.0 if rnn_layers == 1 else 0.5,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise Exception(f"Unknown RNN type: {rnn_type}. Please use either LSTM, GRU or RNN.")

        return RNN

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, int]:

        # if there are no sentences, there is no loss
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        # forward pass to get scores
        scores, gold_labels = self.forward(sentences)  # type: ignore

        # calculate loss given scores and labels
        return self._calculate_loss(scores, gold_labels)

    def forward(self, sentences: Union[List[Sentence], Sentence]):
        """
        Forward propagation through network. Returns gold labels of batch in addition.
        :param sentences: Batch of current sentences
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        self.embeddings.embed(sentences)

        # make a zero-padded tensor for the whole sentence
        lengths, sentence_tensor = self._make_padded_tensor_for_batch(sentences)

        # sort tensor in decreasing order based on lengths of sentences in batch
        sorted_lengths, length_indices = lengths.sort(dim=0, descending=True)
        sentences = [sentences[i] for i in length_indices]
        sentence_tensor = sentence_tensor[length_indices]

        # ----- Forward Propagation -----
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, sorted_lengths, batch_first=True, enforce_sorted=False)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # linear map to tag space
        features = self.linear(sentence_tensor)

        # Depending on whether we are using CRF or a linear layer, scores is either:
        # -- A tensor of shape (batch size, sequence length, tagset size, tagset size) for CRF
        # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
        if self.use_crf:
            features = self.crf(features)
            scores = (features, sorted_lengths, self.crf.transitions)
        else:
            scores = self._get_scores_from_features(features, sorted_lengths)

        # get the gold labels
        gold_labels = self._get_gold_labels(sentences)

        return scores, gold_labels

    def _calculate_loss(self, scores, labels) -> Tuple[torch.Tensor, int]:

        if not any(labels):
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        labels = torch.tensor(
            [
                self.label_dictionary.get_idx_for_item(label[0])
                if len(label) > 0
                else self.label_dictionary.get_idx_for_item("O")
                for label in labels
            ],
            dtype=torch.long,
            device=flair.device,
        )

        return self.loss_function(scores, labels), len(labels)

    def _make_padded_tensor_for_batch(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, torch.Tensor]:
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
            all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )
        return torch.tensor(lengths, dtype=torch.long), sentence_tensor

    @staticmethod
    def _get_scores_from_features(features: torch.Tensor, lengths: torch.Tensor):
        """
        Trims current batch tensor in shape (batch size, sequence length, tagset size) in such a way that all
        pads are going to be removed.
        :param features: torch.tensor containing all features from forward propagation
        :param lengths: length from each sentence in batch in order to trim padding tokens
        """
        features_formatted = []
        for feat, length in zip(features, lengths):
            features_formatted.append(feat[:length])
        scores = torch.cat(features_formatted)

        return scores

    def _get_gold_labels(self, sentences: Union[List[Sentence], Sentence]):
        """
        Extracts gold labels from each sentence.
        :param sentences: List of sentences in batch
        """
        # spans need to be encoded as token-level predictions
        if self.predict_spans:
            all_sentence_labels = []
            for sentence in sentences:
                sentence_labels = ["O"] * len(sentence)
                for label in sentence.get_labels(self.label_type):
                    span: Span = label.data_point
                    if len(span) == 1:
                        sentence_labels[span[0].idx - 1] = "S-" + label.value
                    else:
                        sentence_labels[span[0].idx - 1] = "B-" + label.value
                        sentence_labels[span[-1].idx - 1] = "E-" + label.value
                        for i in range(span[0].idx, span[-1].idx - 1):
                            sentence_labels[i] = "I-" + label.value
                all_sentence_labels.extend(sentence_labels)
            labels = [[label] for label in all_sentence_labels]

        # all others are regular labels for each token
        else:
            labels = [[token.get_label(self.label_type, "O").value] for sentence in sentences for token in sentence]

        return labels

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts labels for current batch with CRF or Softmax.
        :param sentences: List of sentences in batch
        :param mini_batch_size: batch size for test data
        :param return_probabilities_for_all_classes: Whether to return probabilites for all classes
        :param verbose: whether to use progress bar
        :param label_name: which label to predict
        :param return_loss: whether to return loss value
        :param embedding_storage_mode: determines where to store embeddings - can be "gpu", "cpu" or None.
        """
        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            # make sure its a list
            if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
                sentences = [sentences]

            # filter empty sentences
            sentences = [sentence for sentence in sentences if len(sentence) > 0]

            # reverse sort all sequences by their length
            reordered_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)

            if len(reordered_sentences) == 0:
                return sentences

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(reordered_sentences),
                batch_size=mini_batch_size,
            )
            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader, desc="Batch inference")

            overall_loss = torch.zeros(1, device=flair.device)
            batch_no = 0
            label_count = 0
            for batch in dataloader:

                batch_no += 1

                # stop if all sentences are empty
                if not batch:
                    continue

                # get features from forward propagation
                features, gold_labels = self.forward(batch)

                # remove previously predicted labels of this type
                for sentence in batch:
                    sentence.remove_labels(label_name)

                # if return_loss, get loss value
                if return_loss:
                    loss = self._calculate_loss(features, gold_labels)
                    overall_loss += loss[0]
                    label_count += loss[1]

                # Sort batch in same way as forward propagation
                lengths = torch.LongTensor([len(sentence) for sentence in batch])
                _, sort_indices = lengths.sort(dim=0, descending=True)
                batch = [batch[i] for i in sort_indices]

                # make predictions
                if self.use_crf:
                    predictions, all_tags = self.viterbi_decoder.decode(features, return_probabilities_for_all_classes)
                else:
                    predictions, all_tags = self._standard_inference(
                        features, batch, return_probabilities_for_all_classes
                    )

                # add predictions to Sentence
                for sentence, sentence_predictions in zip(batch, predictions):

                    # BIOES-labels need to be converted to spans
                    if self.predict_spans:
                        sentence_tags = [label[0] for label in sentence_predictions]
                        sentence_scores = [label[1] for label in sentence_predictions]
                        predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
                        for predicted_span in predicted_spans:
                            span: Span = sentence[predicted_span[0][0] : predicted_span[0][-1] + 1]
                            span.add_label(label_name, value=predicted_span[2], score=predicted_span[1])

                    # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
                    else:
                        for token, label in zip(sentence.tokens, sentence_predictions):
                            if label[0] in ["O", "_"]:
                                continue
                            token.add_label(typename=label_name, value=label[0], score=label[1])

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _standard_inference(self, features: torch.Tensor, batch: List[Sentence], probabilities_for_all_classes: bool):
        """
        Softmax over emission scores from forward propagation.
        :param features: sentence tensor from forward propagation
        :param batch: list of sentence
        :param probabilities_for_all_classes: whether to return score for each tag in tag dictionary
        """
        softmax_batch = F.softmax(features, dim=1).cpu()
        scores_batch, prediction_batch = torch.max(softmax_batch, dim=1)
        predictions = []
        all_tags = []

        for sentence in batch:
            scores = scores_batch[: len(sentence)]
            predictions_for_sentence = prediction_batch[: len(sentence)]
            predictions.append(
                [
                    (self.label_dictionary.get_item_for_index(prediction), score.item())
                    for token, score, prediction in zip(sentence, scores, predictions_for_sentence)
                ]
            )
            scores_batch = scores_batch[len(sentence) :]
            prediction_batch = prediction_batch[len(sentence) :]

        if probabilities_for_all_classes:
            lengths = [len(sentence) for sentence in batch]
            all_tags = self._all_scores_for_token(softmax_batch, lengths)

        return predictions, all_tags

    def _all_scores_for_token(self, scores: torch.Tensor, lengths: List[int]):
        """
        Returns all scores for each tag in tag dictionary.
        :param scores: Scores for current sentence.
        """
        scores = scores.numpy()
        prob_all_tags = [
            [
                Label(self.label_dictionary.get_item_for_index(score_id), score)
                for score_id, score in enumerate(score_dist)
            ]
            for score_dist in scores
        ]

        prob_tags_per_sentence = []
        previous = 0
        for length in lengths:
            prob_tags_per_sentence.append(prob_all_tags[previous : previous + length])
            previous = length
        return prob_tags_per_sentence

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "tag_dictionary": self.label_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "reproject_embeddings": self.reproject_embeddings,
            "weight_dict": self.weight_dict,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        """Initialize the model from a state dictionary."""
        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = 0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        use_locked_dropout = 0.0 if "use_locked_dropout" not in state.keys() else state["use_locked_dropout"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]

        if state["use_crf"]:
            if "transitions" in state["state_dict"]:
                state["state_dict"]["crf.transitions"] = state["state_dict"]["transitions"]
                del state["state_dict"]["transitions"]

        return super()._init_model_with_state_dict(
            state,
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            hidden_size=state["hidden_size"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            rnn_type=rnn_type,
            reproject_embeddings=reproject_embeddings,
            loss_weights=weights,
            init_from_state_dict=True,
            **kwargs,
        )

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
            "pt-pos-clinical": "/".join(
                [
                    hu_path,
                    "pt-pos-clinical",
                    "pucpr-flair-clinical-pos-tagging-best-model.pt",
                ]
            ),
            # Keyphase models
            "keyphrase": "/".join([hu_path, "keyphrase", "keyphrase-en-scibert.pt"]),
            "negation-speculation": "/".join([hu_path, "negation-speculation", "negation-speculation-model.pt"]),
            # Biomedical models
            "hunflair-paper-cellline": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "cellline",
                    "hunflair-celline-v1.0.pt",
                ]
            ),
            "hunflair-paper-chemical": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "chemical",
                    "hunflair-chemical-v1.0.pt",
                ]
            ),
            "hunflair-paper-disease": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "disease",
                    "hunflair-disease-v1.0.pt",
                ]
            ),
            "hunflair-paper-gene": "/".join([hu_path, "hunflair_smallish_models", "gene", "hunflair-gene-v1.0.pt"]),
            "hunflair-paper-species": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "species",
                    "hunflair-species-v1.0.pt",
                ]
            ),
            "hunflair-cellline": "/".join(
                [
                    hu_path,
                    "hunflair_smallish_models",
                    "cellline",
                    "hunflair-celline-v1.0.pt",
                ]
            ),
            "hunflair-chemical": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-chemical",
                    "hunflair-chemical-full-v1.0.pt",
                ]
            ),
            "hunflair-disease": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-disease",
                    "hunflair-disease-full-v1.0.pt",
                ]
            ),
            "hunflair-gene": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-gene",
                    "hunflair-gene-full-v1.0.pt",
                ]
            ),
            "hunflair-species": "/".join(
                [
                    hu_path,
                    "hunflair_allcorpus_models",
                    "huner-species",
                    "hunflair-species-full-v1.1.pt",
                ]
            ),
        }

        cache_dir = Path("models")

        get_from_model_hub = False

        # check if model name is a valid local file
        if Path(model_name).exists():
            model_path = model_name

        # check if model key is remapped to HF key - if so, print out information
        elif model_name in huggingface_model_map:

            # get mapped name
            hf_model_name = huggingface_model_map[model_name]

            # use mapped name instead
            model_name = hf_model_name
            get_from_model_hub = True

        # if not, check if model key is remapped to direct download location. If so, download model
        elif model_name in hu_model_map:
            model_path = cached_path(hu_model_map[model_name], cache_dir=cache_dir)

        # special handling for the taggers by the @redewiegergabe project (TODO: move to model hub)
        elif model_name == "de-historic-indirect":
            model_file = flair.cache_root / cache_dir / "indirect" / "final-model.pt"
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/indirect.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    flair.cache_root / cache_dir / "indirect.zip",
                    flair.cache_root / cache_dir,
                )
            model_path = str(flair.cache_root / cache_dir / "indirect" / "final-model.pt")

        elif model_name == "de-historic-direct":
            model_file = flair.cache_root / cache_dir / "direct" / "final-model.pt"
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/direct.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    flair.cache_root / cache_dir / "direct.zip",
                    flair.cache_root / cache_dir,
                )
            model_path = str(flair.cache_root / cache_dir / "direct" / "final-model.pt")

        elif model_name == "de-historic-reported":
            model_file = flair.cache_root / cache_dir / "reported" / "final-model.pt"
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/reported.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    flair.cache_root / cache_dir / "reported.zip",
                    flair.cache_root / cache_dir,
                )
            model_path = str(flair.cache_root / cache_dir / "reported" / "final-model.pt")

        elif model_name == "de-historic-free-indirect":
            model_file = flair.cache_root / cache_dir / "freeIndirect" / "final-model.pt"
            if not model_file.exists():
                cached_path(
                    "http://www.redewiedergabe.de/models/freeIndirect.zip",
                    cache_dir=cache_dir,
                )
                unzip_file(
                    flair.cache_root / cache_dir / "freeIndirect.zip",
                    flair.cache_root / cache_dir,
                )
            model_path = str(flair.cache_root / cache_dir / "freeIndirect" / "final-model.pt")

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
            from huggingface_hub import cached_download, hf_hub_url

            url = hf_hub_url(model_name, revision=revision, filename=hf_model_name)

            try:
                model_path = cached_download(
                    url=url,
                    library_name="flair",
                    library_version=flair.__version__,
                    cache_dir=flair.cache_root / "models" / model_folder,
                )
            except HTTPError:
                # output information
                log.error("-" * 80)
                log.error(
                    f"ACHTUNG: The key '{model_name}' was neither found on the ModelHub nor is this a valid path to a file on your system!"
                )
                # log.error(f" - Error message: {e}")
                log.error(" -> Please check https://huggingface.co/models?filter=flair for all available models.")
                log.error(" -> Alternatively, point to a model file on your local drive.")
                log.error("-" * 80)
                Path(flair.cache_root / "models" / model_folder).rmdir()  # remove folder again if not valid

        return model_path

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens.")
        return filtered_sentences

    def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
        for item in dictionary.get_items():
            if item.startswith("B-") or item.startswith("S-") or item.startswith("I-"):
                return True
        return False

    def _print_predictions(self, batch, gold_label_type):

        lines = []
        if self.predict_spans:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                # set predicted token-level
                for predicted_label in datapoint.get_labels("predicted"):
                    predicted_span: Span = predicted_label.data_point
                    prefix = "B-"
                    for token in predicted_span:
                        token.set_label("predicted_bio", prefix + predicted_label.value)
                        prefix = "I-"

                # now print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label('gold_bio').value} "
                        f"{token.get_label('predicted_bio').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")

        else:
            for datapoint in batch:
                # print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label(gold_label_type).value} "
                        f"{token.get_label('predicted').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")
        return lines


class MultiTagger:
    def __init__(self, name_to_tagger: Dict[str, SequenceTagger]):
        super().__init__()
        self.name_to_tagger = name_to_tagger

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        return_loss: bool = False,
        mini_batch_size: int = 32,
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
                logging.warn(
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
                label_name=name,
                return_loss=return_loss,
                embedding_storage_mode="cpu",
                mini_batch_size=mini_batch_size,
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
        models: List[SequenceTagger] = []

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
                        if not embedding.static_embeddings:
                            continue

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
                                model.embeddings.name
                            ]
                            if not previous_embedding.static_embeddings:
                                model.embeddings = previous_embedding
                                break

            taggers[model_name] = model
            models.append(model)

        return cls(taggers)
