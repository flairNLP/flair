import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Label, Sentence, Span, get_spans_from_bio
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path, hf_download
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiDecoder, ViterbiLoss
from flair.training_utils import store_embeddings

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
        loss_weights: Optional[dict[str, float]] = None,
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

    def _init_loss_weights(self, loss_weights: dict[str, float]) -> torch.Tensor:
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

    def forward_loss(self, sentences: list[Sentence]) -> tuple[torch.Tensor, int]:
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

    def _prepare_tensors(self, data_points: Union[list[Sentence], Sentence]) -> tuple[torch.Tensor, torch.LongTensor]:
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

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.LongTensor) -> tuple[torch.Tensor, int]:
        if labels.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        return self.loss_function(scores, labels), len(labels)

    def _make_padded_tensor_for_batch(self, sentences: list[Sentence]) -> tuple[torch.LongTensor, torch.Tensor]:
        names = self.embeddings.get_names()
        lengths: list[int] = [len(sentence.tokens) for sentence in sentences]
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

    def _get_gold_labels(self, sentences: list[Sentence]) -> list[str]:
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
                    if label.value == "O":
                        continue

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

    def _prepare_label_tensor(self, sentences: list[Sentence]):
        gold_labels = self._get_gold_labels(sentences)
        labels = torch.tensor(
            [self.label_dictionary.get_idx_for_item(label) for label in gold_labels],
            dtype=torch.long,
            device=flair.device,
        )
        return labels

    def predict(
        self,
        sentences: Union[list[Sentence], Sentence],
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

            Sentence.set_context_for_sentences(cast(list[Sentence], sentences))

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

    def _standard_inference(self, features: torch.Tensor, batch: list[Sentence], probabilities_for_all_classes: bool):
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

    def _all_scores_for_token(self, sentences: list[Sentence], score_tensor: torch.Tensor, lengths: list[int]):
        """Returns all scores for each tag in tag dictionary."""
        scores = score_tensor.numpy()
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
            "da-ner": "flair/ner-danish",
            "de-ner": "flair/ner-german",
            "de-ler": "flair/ner-german-legal",
            "de-ner-legal": "flair/ner-german-legal",
            "fr-ner": "flair/ner-french",
            "nl-ner": "flair/ner-dutch",
            "ner-ukrainian": "dchaplinsky/flair-uk-ner",
            # Language-specific POS models
            "pos-ukrainian": "dchaplinsky/flair-uk-pos",
            # Historic German
            "de-historic-direct": "aehrm/redewiedergabe-direct",
            "de-historic-indirect": "aehrm/redewiedergabe-indirect",
            "de-historic-reported": "aehrm/redewiedergabe-reported",
            "de-historic-free-indirect": "aehrm/redewiedergabe-freeindirect",
        }

        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"
        hunflair_main_path = hu_path + "/hunflair_allcorpus_models"

        hu_model_map = {
            # English NER models
            "ner-pooled": "/".join([hu_path, "ner-pooled", "en-ner-conll03-pooled-v0.5.pt"]),
            # English SRL models
            "frame-large": "/".join([hu_path, "frame-large", "frame-large.pt"]),
            # Danish models
            "da-pos": "/".join([hu_path, "da-pos", "da-pos-v0.1.pt"]),
            "da-ner": "/".join([hu_path, "NER-danish", "da-ner-v0.1.pt"]),
            # German models
            "de-pos": "/".join([hu_path, "de-pos", "de-pos-ud-hdt-v0.5.pt"]),
            "de-pos-tweets": "/".join([hu_path, "de-pos-tweets", "de-pos-twitter-v0.1.pt"]),
            "de-ner": "/".join([hu_path, "de-ner", "de-ner-conll03-v0.4.pt"]),
            "de-ner-germeval": "/".join([hu_path, "de-ner-germeval", "de-ner-germeval-0.4.1.pt"]),
            # Arabic models
            "ar-ner": "/".join([hu_path, "arabic", "ar-ner.pt"]),
            "ar-pos": "/".join([hu_path, "arabic", "ar-pos.pt"]),
            # French models
            "fr-ner": "/".join([hu_path, "fr-ner", "fr-ner-wikiner-0.4.pt"]),
            # Dutch models
            "nl-ner": "/".join([hu_path, "nl-ner", "nl-ner-bert-conll02-v0.8.pt"]),
            "nl-ner-rnn": "/".join([hu_path, "nl-ner-rnn", "nl-ner-conll02-v0.14.0.pt"]),
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
            "negation-speculation": "/".join([hu_path, "negation-speculation-v14", "negation-speculation-v0.14.0.pt"]),
            # Biomedical models
            "hunflair-cellline": "/".join([hunflair_main_path, "huner-cellline", "hunflair-cellline.pt"]),
            "hunflair-chemical": "/".join([hunflair_main_path, "huner-chemical", "hunflair-chemical.pt"]),
            "hunflair-disease": "/".join([hunflair_main_path, "huner-disease", "hunflair-disease.pt"]),
            "hunflair-gene": "/".join([hunflair_main_path, "huner-gene", "hunflair-gene.pt"]),
            "hunflair-species": "/".join([hunflair_main_path, "huner-species", "hunflair-species.pt"]),
        }

        cache_dir = Path("models")

        # check if model name is a valid local file
        if Path(model_name).exists():
            model_path = model_name

        # check if model key is remapped to HF key - if so, print out information
        elif model_name in huggingface_model_map:
            # get mapped name
            hf_model_name = huggingface_model_map[model_name]

            model_path = hf_download(hf_model_name)

        # if not, check if model key is remapped to direct download location. If so, download model
        elif model_name in hu_model_map:
            model_path = cached_path(hu_model_map[model_name], cache_dir=cache_dir)

            if model_name.startswith("hunflair-"):
                log.warning(
                    "HunFlair (version 1) is deprecated. Consider using HunFlair2 for improved extraction performance: "
                    "Classifier.load('hunflair2')."
                    "See https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR2.md for further "
                    "information."
                )

        # for all other cases (not local file or special download location), use HF model hub
        else:
            model_path = hf_download(model_name)

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
    def _filter_empty_sentences(sentences: list[Sentence]) -> list[Sentence]:
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
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "SequenceTagger":
        from typing import cast

        return cast("SequenceTagger", super().load(model_path=model_path))
