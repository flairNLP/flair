import itertools
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from urllib.error import HTTPError

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Label, Sentence, Span, get_spans_from_bio, DT
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import TokenEmbeddings, TransformerWordEmbeddings
from flair.file_utils import cached_path, unzip_file, Tqdm
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiDecoder, ViterbiLoss
from flair.training_utils import store_embeddings, Result

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
        loss_weights: Optional[Dict[str, float]] = None,
        init_from_state_dict: bool = False,
        allow_unk_predictions: bool = False,
    ) -> None:
        """Sequence Tagger class for predicting labels for single tokens. Can be parameterized by several attributes.

        In case of multitask learning, pass shared embeddings or shared rnn into respective attributes.

        Args:
            embeddings: Embeddings to use during training and prediction
            tag_dictionary: Dictionary containing all tags from corpus which can be predicted
            tag_type: type of tag which is going to be predicted in case a corpus has multiple annotations
            use_rnn: If true, use a RNN, else Linear layer.
            rnn: Takes a torch.nn.Module as parameter by which you can pass a shared RNN between different tasks.
            rnn_type: Specifies the RNN type to use, default is 'LSTM', can choose between 'GRU' and 'RNN' as well.
            hidden_size: Hidden size of RNN layer
            rnn_layers: number of RNN layers
            bidirectional: If True, RNN becomes bidirectional
            use_crf: If True, use a Conditional Random Field for prediction, else linear map to tag space.
            reproject_embeddings: If True, add a linear layer on top of embeddings, if you want to imitate fine tune non-trainable embeddings.
            dropout: If > 0, then use dropout.
            word_dropout: If > 0, then use word dropout.
            locked_dropout: If > 0, then use locked dropout.
            train_initial_hidden_state: if True, trains initial hidden state of RNN
            loss_weights: Dictionary of weights for labels for the loss function. If any label's weight is unspecified it will default to 1.0.
            init_from_state_dict: Indicator whether we are loading a model from state dict since we need to transform previous models' weights into CRF instance weights
            allow_unk_predictions: If True, allows spans to predict <unk> too.
            tag_format: the format to encode spans as tags, either "BIO" or "BIOES"
        """
        super().__init__()

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
                assert self.tag_format in ["BIOES", "BIO"]
                for label in tag_dictionary.get_items():
                    if label == "<unk>":
                        continue
                    self.label_dictionary.add_item("O")
                    if self.tag_format == "BIOES":
                        self.label_dictionary.add_item("S-" + label)
                        self.label_dictionary.add_item("B-" + label)
                        self.label_dictionary.add_item("E-" + label)
                        self.label_dictionary.add_item("I-" + label)
                    if self.tag_format == "BIO":
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

        # remove word dropout if there is no contact over the sequence dimension.
        if not use_crf and not use_rnn:
            word_dropout = 0.0

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
            self.train_initial_hidden_state = False

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
        """Initializes the loss weights based on given dictionary.

        Args:
            loss_weights: dictionary - contains loss weights
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1.0 for _ in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights:
                weight_list[i] = loss_weights[tag]

        return torch.tensor(weight_list).to(flair.device)

    def _init_initial_hidden_state(self, num_directions: int):
        """Initializes hidden states given the number of directions in RNN.

        Args:
            num_directions: Number of directions in RNN.
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
        """Static wrapper function returning an RNN instance from PyTorch.

        Args:
            rnn_type: Type of RNN from torch.nn
            rnn_layers: number of layers to include
            hidden_size: hidden size of RNN cell
            bidirectional: If True, RNN cell is bidirectional
            rnn_input_dim: Input dimension to RNN cell
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

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        # if there are no sentences, there is no loss
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
        sentences = sorted(sentences, key=len, reverse=True)
        gold_labels = self._prepare_label_tensor(sentences)
        sentence_tensor, lengths = self._prepare_tensors(sentences)

        # forward pass to get scores
        scores = self.forward(sentence_tensor, lengths)

        # calculate loss given scores and labels
        return self._calculate_loss(scores, gold_labels)

    def _prepare_tensors(self, data_points: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, torch.LongTensor]:
        sentences = [data_points] if not isinstance(data_points, list) else data_points
        self.embeddings.embed(sentences)

        # make a zero-padded tensor for the whole sentence
        lengths, sentence_tensor = self._make_padded_tensor_for_batch(sentences)

        return sentence_tensor, lengths

    def forward(self, sentence_tensor: torch.Tensor, lengths: torch.LongTensor):
        """Forward propagation through network.

        Args:
            sentence_tensor: A tensor representing the batch of sentences.
            lengths: A IntTensor representing the lengths of the respective sentences.
        """
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, lengths, batch_first=True)
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
            scores = (features, lengths, self.crf.transitions)
        else:
            scores = self._get_scores_from_features(features, lengths)

        return scores

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.LongTensor) -> Tuple[torch.Tensor, int]:
        if labels.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        return self.loss_function(scores, labels), len(labels)

    def _make_padded_tensor_for_batch(self, sentences: List[Sentence]) -> Tuple[torch.LongTensor, torch.Tensor]:
        names = self.embeddings.get_names()
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=self.linear.weight.dtype,
            device=flair.device,
        )
        all_embs = []
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
        return torch.LongTensor(lengths), sentence_tensor

    @staticmethod
    def _get_scores_from_features(features: torch.Tensor, lengths: torch.Tensor):
        """Remove paddings to get a smaller tensor.

        Trims current batch tensor in shape (batch size, sequence length, tagset size)
        in such a way that all pads are going to be removed.

        Args:
            features: all features from forward propagation
            lengths: length from each sentence in batch in order to trim padding tokens
        """
        features_formatted = []
        for feat, length in zip(features, lengths):
            features_formatted.append(feat[:length])
        scores = torch.cat(features_formatted)

        return scores

    def _get_gold_labels(self, sentences: List[Sentence]) -> List[str]:
        """Extracts gold labels from each sentence.

        Args:
            sentences: List of sentences in batch
        """
        # spans need to be encoded as token-level predictions
        if self.predict_spans:
            all_sentence_labels = []
            for sentence in sentences:
                sentence_labels = ["O"] * len(sentence)
                for label in sentence.get_labels(self.label_type):
                    span: Span = label.data_point
                    if self.tag_format == "BIOES":
                        if len(span) == 1:
                            sentence_labels[span[0].idx - 1] = "S-" + label.value
                        else:
                            sentence_labels[span[0].idx - 1] = "B-" + label.value
                            sentence_labels[span[-1].idx - 1] = "E-" + label.value
                            for i in range(span[0].idx, span[-1].idx - 1):
                                sentence_labels[i] = "I-" + label.value
                    else:
                        sentence_labels[span[0].idx - 1] = "B-" + label.value
                        for i in range(span[0].idx, span[-1].idx):
                            sentence_labels[i] = "I-" + label.value
                all_sentence_labels.extend(sentence_labels)
            labels = all_sentence_labels

        # all others are regular labels for each token
        else:
            labels = [token.get_label(self.label_type, "O").value for sentence in sentences for token in sentence]

        return labels

    def _prepare_label_tensor(self, sentences: List[Sentence]):
        gold_labels = self._get_gold_labels(sentences)
        labels = torch.tensor(
            [self.label_dictionary.get_idx_for_item(label) for label in gold_labels],
            dtype=torch.long,
            device=flair.device,
        )
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
        force_token_predictions: bool = False,
    ):
        """Predicts labels for current batch with CRF or Softmax.

        Args:
            sentences: List of sentences in batch
            mini_batch_size: batch size for test data
            return_probabilities_for_all_classes: Whether to return probabilities for all classes
            verbose: whether to use progress bar
            label_name: which label to predict
            return_loss: whether to return loss value
            embedding_storage_mode: determines where to store embeddings - can be "gpu", "cpu" or None.
            force_token_predictions: add labels per token instead of span labels, even if `self.predict_spans` is True
        """
        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            # make sure it's a list
            if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
                sentences = [sentences]

            Sentence.set_context_for_sentences(cast(List[Sentence], sentences))

            # filter empty sentences
            sentences = [sentence for sentence in sentences if len(sentence) > 0]

            # reverse sort all sequences by their length
            reordered_sentences = sorted(sentences, key=len, reverse=True)

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
            label_count = 0
            for batch in dataloader:
                # stop if all sentences are empty
                if not batch:
                    continue

                # get features from forward propagation
                sentence_tensor, lengths = self._prepare_tensors(batch)
                features = self.forward(sentence_tensor, lengths)

                # remove previously predicted labels of this type
                for sentence in batch:
                    sentence.remove_labels(label_name)

                # if return_loss, get loss value
                if return_loss:
                    gold_labels = self._prepare_label_tensor(batch)
                    loss = self._calculate_loss(features, gold_labels)
                    overall_loss += loss[0]
                    label_count += loss[1]

                # make predictions
                if self.use_crf:
                    predictions, all_tags = self.viterbi_decoder.decode(
                        features, return_probabilities_for_all_classes, batch
                    )
                else:
                    predictions, all_tags = self._standard_inference(
                        features, batch, return_probabilities_for_all_classes
                    )

                # add predictions to Sentence
                for sentence, sentence_predictions in zip(batch, predictions):
                    # BIOES-labels need to be converted to spans
                    if self.predict_spans and not force_token_predictions:
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
                for sentence, sent_all_tags in zip(batch, all_tags):
                    for token, token_all_tags in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count
            return None

    def _standard_inference(self, features: torch.Tensor, batch: List[Sentence], probabilities_for_all_classes: bool):
        """Softmax over emission scores from forward propagation.

        Args:
            features: sentence tensor from forward propagation
            batch: sentences
            probabilities_for_all_classes: whether to return score for each tag in tag dictionary
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
            all_tags = self._all_scores_for_token(batch, softmax_batch, lengths)

        return predictions, all_tags

    def _all_scores_for_token(self, sentences: List[Sentence], scores: torch.Tensor, lengths: List[int]):
        """Returns all scores for each tag in tag dictionary."""
        scores = scores.numpy()
        tokens = [token for sentence in sentences for token in sentence]
        prob_all_tags = [
            [
                Label(token, self.label_dictionary.get_item_for_index(score_id), score)
                for score_id, score in enumerate(score_dist)
            ]
            for score_dist, token in zip(scores, tokens)
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
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "hidden_size": self.hidden_size,
            "tag_dictionary": self.label_dictionary,
            "tag_format": self.tag_format,
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
            "train_initial_hidden_state": self.train_initial_hidden_state,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        if state["use_crf"] and "transitions" in state["state_dict"]:
            state["state_dict"]["crf.transitions"] = state["state_dict"]["transitions"]
            del state["state_dict"]["transitions"]

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            tag_dictionary=state.get("tag_dictionary"),
            tag_format=state.get("tag_format", "BIOES"),
            tag_type=state.get("tag_type"),
            use_crf=state.get("use_crf"),
            use_rnn=state.get("use_rnn"),
            rnn_layers=state.get("rnn_layers"),
            hidden_size=state.get("hidden_size"),
            dropout=state.get("use_dropout", 0.0),
            word_dropout=state.get("use_word_dropout", 0.0),
            locked_dropout=state.get("use_locked_dropout", 0.0),
            rnn_type=state.get("rnn_type", "LSTM"),
            reproject_embeddings=state.get("reproject_embeddings", True),
            loss_weights=state.get("weight_dict"),
            init_from_state_dict=True,
            train_initial_hidden_state=state.get("train_initial_hidden_state", False),
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
            "ner-ukrainian": "dchaplinsky/flair-uk-ner",
            # Language-specific POS models
            "pos-ukrainian": "dchaplinsky/flair-uk-pos",
        }

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"
        hunflair_paper_path = hu_path + "/hunflair_smallish_models"
        hunflair_main_path = hu_path + "/hunflair_allcorpus_models"

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
            "frame-large": "/".join([hu_path, "frame-large", "frame-large.pt"]),
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
            "hunflair-paper-cellline": "/".join([hunflair_paper_path, "cellline", "hunflair-celline-v1.0.pt"]),
            "hunflair-paper-chemical": "/".join([hunflair_paper_path, "chemical", "hunflair-chemical-v1.0.pt"]),
            "hunflair-paper-disease": "/".join([hunflair_paper_path, "disease", "hunflair-disease-v1.0.pt"]),
            "hunflair-paper-gene": "/".join([hunflair_paper_path, "gene", "hunflair-gene-v1.0.pt"]),
            "hunflair-paper-species": "/".join([hunflair_paper_path, "species", "hunflair-species-v1.0.pt"]),
            "hunflair-cellline": "/".join([hunflair_main_path, "cellline", "hunflair-celline-v1.0.pt"]),
            "hunflair-chemical": "/".join([hunflair_main_path, "huner-chemical", "hunflair-chemical-full-v1.0.pt"]),
            "hunflair-disease": "/".join([hunflair_main_path, "huner-disease", "hunflair-disease-full-v1.0.pt"]),
            "hunflair-gene": "/".join([hunflair_main_path, "huner-gene", "hunflair-gene-full-v1.0.pt"]),
            "hunflair-species": "/".join([hunflair_main_path, "huner-species", "hunflair-species-full-v1.1.pt"]),
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
            model_folder = model_name.split("/", maxsplit=1)[1] if "/" in model_name else model_name

            # Lazy import
            from huggingface_hub.file_download import hf_hub_download

            try:
                model_path = hf_hub_download(
                    repo_id=model_name,
                    filename=hf_model_name,
                    revision=revision,
                    library_name="flair",
                    library_version=flair.__version__,
                    cache_dir=flair.cache_root / "models" / model_folder,
                )
            except HTTPError:
                # output information
                log.error("-" * 80)
                log.error(
                    f"ERROR: The key '{model_name}' was neither found on the ModelHub nor is this a valid path to a file on your system!"
                )
                log.error(" -> Please check https://huggingface.co/models?filter=flair for all available models.")
                log.error(" -> Alternatively, point to a model file on your local drive.")
                log.error("-" * 80)
                Path(flair.cache_root / "models" / model_folder).rmdir()  # remove folder again if not valid
                raise

        return model_path

    def _generate_model_card(self, repo_id):
        return f"""---
tags:
- flair
- token-classification
- sequence-tagger-model
---

### Demo: How to use in Flair

Requires:
- **[Flair](https://github.com/flairNLP/flair/)** (`pip install flair`)

```python
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("{repo_id}")

# make example sentence
sentence = Sentence("On September 1st George won 1 dollar while watching Game of Thrones.")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
```"""

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: Optional[bool] = None,
        commit_message: str = "Add new SequenceTagger model.",
    ):
        """Uploads the Sequence Tagger model to a Hugging Face Hub repository.

        Args:
            repo_id: A namespace (user or an organization) and a repo name separated by a `/`.
            token: An authentication token (See https://huggingface.co/settings/token).
            private: Whether the repository is private.
            commit_message: Message to commit while pushing.

        Returns: The url of the repository.
        """
        # Lazy import
        from huggingface_hub import create_repo, model_info, upload_folder

        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save model weight
            local_model_path = tmp_path / "pytorch_model.bin"
            self.save(local_model_path)

            # Determine if model card already exists
            info = model_info(repo_id, use_auth_token=token)
            write_readme = all(f.rfilename != "README.md" for f in info.siblings)

            # Generate and save model card
            if write_readme:
                model_card_content = self._generate_model_card(repo_id)
                readme_path = tmp_path / "README.md"
                with readme_path.open("w", encoding="utf-8") as f:
                    f.write(model_card_content)

            # Upload files
            upload_folder(
                repo_id=repo_id,
                folder_path=tmp_path,
                path_in_repo="",
                token=token,
                commit_message=commit_message,
            )
            return repo_url

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens.")
        return filtered_sentences

    def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
        return any(item.startswith(("B-", "S-", "I-")) for item in dictionary.get_items())

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
    
    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "SequenceTagger":
        from typing import cast

        return cast("SequenceTagger", super().load(model_path=model_path))


class EarlyExitSequenceTagger(SequenceTagger):
    def __init__(
        self,
        embeddings: TransformerWordEmbeddings, # layer_mean = False, layers = "all"
        tag_dictionary: Dictionary,
        tag_type: str,
        use_rnn = False,
        use_crf = False, 
        reproject_embeddings = False,
        weighted_loss: bool = True,
        last_layer_only: bool = False,
        print_all_predictions = True,
        modified_loss = False,
        relabel_noisy = False,
        **seqtaggerargs
    ):
        """
        Adds Early-Exit functionality to the SequenceTagger
        :param weighted_loss: controls whether to compute a weighted or a simple average loss
        over all the early-exit layers.
        :param last_layer_only: allows to use outputs of the last layer only to train the
        model (like in the case of the regular SequenceTagger).
        """
        super().__init__(
            embeddings = embeddings,
            tag_dictionary = tag_dictionary,
            tag_type = tag_type,
            use_rnn = use_rnn,
            use_crf = use_crf, 
            reproject_embeddings = reproject_embeddings,
            **seqtaggerargs
        )
        
        if embeddings.layer_mean:
            raise AssertionError("layer_mean must be disabled for the transformer embeddings")
        self.n_layers = len(embeddings.layer_indexes) # the output of the emb layer before the transformer blocks counts as well
        self.final_embedding_size = int(embeddings.embedding_length / self.n_layers)
        self.linear = torch.nn.ModuleList(
            torch.nn.Linear(self.final_embedding_size, len(self.label_dictionary)) 
            for _ in range(self.n_layers)
            )
        self.weighted_loss = weighted_loss
        self.last_layer_only = last_layer_only
        self.print_all_predictions = print_all_predictions
        self.modified_loss=modified_loss
        if self.modified_loss:
            self.loss_function = (
                ViterbiLoss(self.label_dictionary)
                if use_crf
                else torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction='none')
            )
        self.to(flair.device)

    def _make_padded_tensor_for_batch(self, sentences: List[Sentence]) -> Tuple[torch.LongTensor, torch.Tensor]:
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
                self.n_layers,
                self.final_embedding_size,
            ]
        )
        return torch.LongTensor(lengths), sentence_tensor

    def forward(self, sentence_tensor: torch.Tensor, lengths: torch.LongTensor):  # type: ignore[override]
        """
        Forward propagation through network.
        :param sentence_tensor: A tensor representing the batch of sentences.
        :param lengths: A IntTensor representing the lengths of the respective sentences.
        """
        scores = []
        for i in range(self.n_layers):
            sentence_layer_tensor = sentence_tensor[:, :, i, :]
            if self.use_dropout:
                sentence_layer_tensor = self.dropout(sentence_layer_tensor)
            if self.use_word_dropout:
                sentence_layer_tensor = self.word_dropout(sentence_layer_tensor)
            if self.use_locked_dropout:
                sentence_layer_tensor = self.locked_dropout(sentence_layer_tensor)

            # linear map to tag space
            features = self.linear[i](sentence_layer_tensor)

            # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
            layer_scores = self._get_scores_from_features(features, lengths)
            scores.append(layer_scores)

        return torch.stack(scores)

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.LongTensor) -> Tuple[torch.Tensor, int]:

        if labels.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        if self.last_layer_only:
            loss = self.loss_function(scores[-1], labels)
        elif self.modified_loss: 
            if self.weighted_loss:
                
                layer_weights = torch.arange(self.n_layers, device=flair.device)
                layer_weighted_loss = 0
                for i in range(self.n_layers):
                    layer_loss = self.loss_function(scores[i], labels)
                    layer_weighted_loss += layer_weights[i] * layer_loss
                loss = layer_weighted_loss / sum(layer_weights) #per-sample layer-weighted average loss
            else: 
                loss = 0
                for i in range(1, self.n_layers):
                    loss += self.loss_function(scores[i], labels)
                loss = loss / (self.n_layers - 1) #per-sample layer average loss
            
            softmax_batch = F.softmax(scores, dim=2).detach()
            
            last_layer_prediction = torch.argmax(softmax_batch[-1,:,:], dim=-1)

            max_pd_tensor  = torch.full(loss.size(), self.n_layers, device=flair.device, requires_grad=False)

            pds = []
            for i in range(softmax_batch.size()[1]):
                pd = self._calculate_pd(softmax_batch[:, i, :])
                pds.append(pd+1)
            pds = torch.tensor(pds, device = flair.device, requires_grad=False) # get per-sample PDs

            correct_prediction_indicator = torch.eq(last_layer_prediction, labels).int()
            incorrect_prediction_indicator = torch.ones_like(correct_prediction_indicator) - correct_prediction_indicator

            # per-sample loss - correct predictions weighted by PD
            loss_correct = (max_pd_tensor * loss) / pds * correct_prediction_indicator # correct predictions -> downweigh high PD
            # per-sample loss - incorrect predictions weighted by PD
            loss_incorrect = (pds * loss) / max_pd_tensor * incorrect_prediction_indicator # incorrect predictions -> downweigh low PD

            loss = loss_correct + loss_incorrect
            loss = loss.sum() #sample-average loss
        else:
            if self.weighted_loss:
                layer_weights = torch.arange(self.n_layers, device=flair.device)
                
                # 0.01 and 1 weights
                #layer_weights = [0.01 for i in range(self.n_layers)]
                #layer_weights[-1] = 1
                #layer_weights = torch.tensor(layer_weights, dtype=torch.float, device=flair.device, requires_grad=False)

                layer_weighted_loss = 0
                for i in range(self.n_layers):
                    layer_loss = self.loss_function(scores[i], labels)
                    layer_weighted_loss += layer_weights[i] * layer_loss
                loss = layer_weighted_loss / sum(layer_weights) # sample-sum layer-weighted average loss
            else:
                loss = 0
                for i in range(1, self.n_layers):
                    loss += self.loss_function(scores[i], labels)
                loss = loss / (self.n_layers - 1) #sample-sum layer average loss
        return loss, len(labels)

    def _calculate_pd(self, scores: torch.Tensor, label_threshold=None) -> int:
        """
        Calculates the prediction depth for a given (single) data point.
        :param scores: tensor with softmax or sigmoid scores of all layers
        :param label_threshold: relevant only for multi-label classification
        """
        pd = self.n_layers - 1

        pred_labels = torch.argmax(scores, dim=-1)
        for i in range(self.n_layers - 2, -1, -1):  # iterate over the layers starting from the penultimate one
            if pred_labels[i] == pred_labels[-1]:
                pd -= 1
            else:
                break
        return pd       

    def _standard_inference(self, features: torch.Tensor, batch: List[Sentence], probabilities_for_all_classes: bool):
        """
        Softmax over emission scores from forward propagation.
        :param features: sentence tensor from forward propagation
        :param batch: list of sentence
        :param probabilities_for_all_classes: whether to return score for each tag in tag dictionary
        """
        softmax_batch = F.softmax(features, dim=2).cpu()
        full_scores_batch, full_prediction_batch = torch.max(softmax_batch, dim=2)
        predictions = []
        all_tags = []

        for i in range(self.n_layers):
            layer_predictions = []
            scores_batch, prediction_batch = full_scores_batch[i], full_prediction_batch[i]
            for sentence in batch:
                scores = scores_batch[: len(sentence)]
                predictions_for_sentence = prediction_batch[: len(sentence)]
                layer_predictions.append(
                    [
                        (self.label_dictionary.get_item_for_index(prediction), score.item())
                        for token, score, prediction in zip(sentence, scores, predictions_for_sentence)
                    ]
                )
                scores_batch = scores_batch[len(sentence) :]
                prediction_batch = prediction_batch[len(sentence) :]
            predictions.append(layer_predictions)
        
        new_sentence_start = 0

        for sentence in batch:
            for k, token in enumerate(sentence):
                pd = self._calculate_pd(softmax_batch[:, new_sentence_start + k, :])
                if not token.has_label('PD'):
                    token.add_label(typename="PD", value="PD", score=pd)
                else:
                    token.set_label(typename="PD", value="PD", score=pd)

            new_sentence_start += len(sentence)

            #TODO: also log overall PD for sentence


        if probabilities_for_all_classes:
            for i in range(self.n_layers):
                lengths = [len(sentence) for sentence in batch]
                layer_tags = self._all_scores_for_token(batch, softmax_batch[i], lengths)
                all_tags.append(layer_tags)

        return predictions, all_tags


    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        force_token_predictions: bool = False,
        layer_idx: int = -1,
    ):  # type: ignore
        """
        Predicts labels for current batch with Softmax.
        :param sentences: List of sentences in batch
        :param mini_batch_size: batch size for test data
        :param return_probabilities_for_all_classes: Whether to return probabilites for all classes
        :param verbose: whether to use progress bar
        :param label_name: which label to predict
        :param return_loss: whether to return loss value
        :param embedding_storage_mode: determines where to store embeddings - can be "gpu", "cpu" or None.
        :param layer_idx: determines which layer is used to write the predictions to spans or tokens.
        """
        if abs(layer_idx) > self.n_layers:
            raise ValueError('Layer index out of range')

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
            reordered_sentences = sorted(sentences, key=len, reverse=True)

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
            label_count = 0
            for batch in dataloader:

                # stop if all sentences are empty
                if not batch:
                    continue

                # get features from forward propagation
                sentence_tensor, lengths = self._prepare_tensors(batch)
                features = self.forward(sentence_tensor, lengths)

                # remove previously predicted labels of this type
                for sentence in batch:
                    sentence.remove_labels(label_name)

                # if return_loss, get loss value
                if return_loss:
                    gold_labels = self._prepare_label_tensor(batch)
                    loss = self._calculate_loss(features, gold_labels)
                    overall_loss += loss[0]
                    label_count += loss[1]

                # make predictions
                predictions, all_tags = self._standard_inference(
                    features, batch, return_probabilities_for_all_classes
                )

                # add predictions to Sentence
                for sentence, sentence_predictions in zip(batch, predictions[layer_idx]):

                    # BIOES-labels need to be converted to spans
                    if self.predict_spans and not force_token_predictions:
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
                if len(all_tags) > 0:
                    for (sentence, sent_all_tags) in zip(batch, all_tags[layer_idx]):
                        for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                            token.add_tags_proba_dist(label_name, token_all_tags)

                store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count


    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        layer_idx = -1,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        """
        This override contains solely the following chagne:
        :param layer_idx: determines which layer is used to write the predictions to spans or tokens.
        This parameters is passed onto the :predict: method to allow for the evaluation of each early-exit
        layer individually.
        """

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}

            loader = DataLoader(data_points, batch_size=mini_batch_size)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=return_loss,
                    layer_idx=layer_idx,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count

                # get the gold labels
                for datapoint in batch:

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    for predicted_span in datapoint.get_labels("predicted"):
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                        else:
                            all_predicted_values[representation].append(predicted_span.value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path and layer_idx==-1 and self.print_all_predictions:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"]
                )

            # write all_predicted_values to out_file if set (per-epoch)
            if out_path and layer_idx==-1 and self.print_all_predictions:
                epoch_log_path = str(out_path)[:-4]+'_'+str(self.model_card["training_parameters"]["epoch"])+'.tsv'
                with open(Path(epoch_log_path), "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ]
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]

        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            "\nResults:"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}"
            f"\n- Accuracy {accuracy_score}"
            "\n\nBy class:\n" + classification_report
        )

        if average_over > 0:
            eval_loss /= average_over

        result = Result(
            main_score=main_score,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            scores={'loss':eval_loss.item()},
        )

        return result


    def _print_predictions(self, batch, gold_label_type):
        # this override also prints out PD for each token
        lines = []
        if self.predict_spans:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("clean_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                sentence_flag = datapoint.get_labels(gold_label_type) != datapoint.get_labels(gold_label_type+'_clean')

                # set clean token-level
                for clean_label in datapoint.get_labels(gold_label_type+'_clean'):
                    clean_span: Span = clean_label.data_point
                    prefix = "B-"
                    for token in clean_span:
                        token.set_label("clean_bio", prefix + clean_label.value) # TODO: add checks, this only works if ner_clean column is given
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
                    gold = token.get_label('gold_bio').value
                    clean = token.get_label('clean_bio').value
                    pred = token.get_label('predicted_bio').value
                    eval_line = (
                        f"{token.text} "
                        f"{gold} " # observed (noisy) label
                        f"{clean} " # clean label
                        f"{pred} " # predicted label
                        f"{pred == gold} " # correct prediction flag
                        f"{gold != clean} " # noisy flag 
                        f"{sentence_flag} " # sentence noisy flag
                        f"{token.get_label('PD').score}\n"
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
                        f"{token.get_label('predicted').value} "
                        f"{token.get_label('PD').score}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")

        return lines
 
    


class DetachedEarlyExitSequenceTagger(EarlyExitSequenceTagger):
    def __init__(
        self,
        modify_last_decoder_lr = True,
        **seqtaggerargs
    ):
        super().__init__(
            **seqtaggerargs
        )
        self.modify_last_decoder_lr = modify_last_decoder_lr

    def forward(self, sentence_tensor: torch.Tensor, lengths: torch.LongTensor):  # type: ignore[override]
        """
        Forward propagation through network.
        :param sentence_tensor: A tensor representing the batch of sentences.
        :param lengths: A IntTensor representing the lengths of the respective sentences.
        """
        scores = []
        for i in range(self.n_layers):
            sentence_layer_tensor = sentence_tensor[:, :, i, :]
            if self.use_dropout:
                sentence_layer_tensor = self.dropout(sentence_layer_tensor)
            if self.use_word_dropout:
                sentence_layer_tensor = self.word_dropout(sentence_layer_tensor)
            if self.use_locked_dropout:
                sentence_layer_tensor = self.locked_dropout(sentence_layer_tensor)

            # linear map to tag space
            if i < self.n_layers-1:
                features = self.linear[i](sentence_layer_tensor.detach()) # if any layer but last .detach()
            else:
                features = self.linear[i](sentence_layer_tensor)
            # think about having a factor for all other decoders

            # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
            layer_scores = self._get_scores_from_features(features, lengths)
            scores.append(layer_scores)

        return torch.stack(scores)


class HybridEarlyExitSequenceTagger(EarlyExitSequenceTagger):
    def __init__(
        self,
        embeddings: TransformerWordEmbeddings, # layer_mean = False, layers = "all"
        warm_up_epochs: bool = False,
        **seqtaggerargs
    ):
        """
        Modify Early-Exit SequenceTagger
        :param warm_up_epochs: number of epochs to fine-tune the TransformerEmbeddings, together with the last Linear layer, 
                            after that they are frozen and only the N linear layers are fine-tuned for the remaining epochs.
        """

        self.warm_up_epochs = warm_up_epochs

        super().__init__(
            embeddings = embeddings,
            last_layer_only = True,
            **seqtaggerargs
        )

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        layer_idx = -1,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        """
        This override contains solely the following chagne:
        when evaluate is 
        
        """

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}

            loader = DataLoader(data_points, batch_size=mini_batch_size)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=return_loss,
                    layer_idx=layer_idx,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count

                # get the gold labels
                for datapoint in batch:

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    for predicted_span in datapoint.get_labels("predicted"):
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                        else:
                            all_predicted_values[representation].append(predicted_span.value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path and layer_idx==-1:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"]
                )

            # write all_predicted_values to out_file if set
            if out_path and layer_idx==-1:
                epoch_log_path = str(out_path)[:-4]+'_'+str(self.model_card["training_parameters"]["epoch"])+'.tsv'
                with open(Path(epoch_log_path), "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ]
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]

        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            "\nResults:"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}"
            f"\n- Accuracy {accuracy_score}"
            "\n\nBy class:\n" + classification_report
        )

        if average_over > 0:
            eval_loss /= average_over

        result = Result(
            main_score=main_score,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            scores={'loss':eval_loss.item()},
        )

        ## this would only work if evaluate is called at the end of each epoch, if not it needs to be modified
        if self.embeddings.fine_tune and self.model_card["training_parameters"]["epoch"] == self.warm_up_epochs:
            
            # change training mode: freeze embeddings, enable fine-tuning of all N classifier heads
            self.last_layer_only = False
            self.embeddings.fine_tune =  False

        return result
