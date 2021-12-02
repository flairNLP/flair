import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Union, List
from urllib.error import HTTPError

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

import flair.nn
from flair.data import Sentence, Dictionary, Label, DataPoint, Token
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.training_utils import store_embeddings
from flair.file_utils import cached_path, unzip_file

from .sequence_tagger_utils.crf import CRF
from .sequence_tagger_utils.viterbi import ViterbiLoss, ViterbiDecoder
from .sequence_tagger_utils.utils import START_TAG, STOP_TAG
from ..datasets import DataLoader, SentenceDataset

log = logging.getLogger("flair")


class SequenceTagger(flair.nn.DefaultClassifier):

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_rnn: bool = True,
            rnn: Optional[torch.nn.Module] = None,
            rnn_type: str = "LSTM",
            hidden_size: int = 256,
            rnn_layers: int = 1,
            bidirectional: bool = True,
            use_crf: bool = True,
            reproject_embeddings: bool = True,
            dropout: float = 0.0,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.5,
            train_initial_hidden_state: bool = False,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            init_from_state_dict: bool = False
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
        :param beta: Beta value for evaluation metric.
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        :param init_from_state_dict: Indicator whether we are loading a model from state dict
            since we need to transform previous models' weights into CRF instance weights
        """
        super(SequenceTagger, self).__init__(label_dictionary=tag_dictionary)

        # ----- Embedding specific parameters -----
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.tag_type = tag_type

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
        if use_crf \
                and not {START_TAG.encode(), STOP_TAG.encode()}.issubset(self.tag_dictionary.item2idx.keys())\
                and not init_from_state_dict:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)
            self.tagset_size += 2

        # ----- Dropout parameters -----
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout = True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # ----- F-beta score for training + annealing
        self.beta = beta

        # ----- Model layers -----
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        # ----- Dropout layers -----
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        # ----- RNN layer -----
        if use_rnn:
            # If shared RNN provided, else create one for model
            if rnn:
                self.rnn = rnn
            else:
                self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim=embedding_dim)
            num_directions = 2 if self.bidirectional else 1
            hidden_output_dim = self.rnn.hidden_size * num_directions

            # Whether to train initial hidden state
            self.train_initial_hidden_state = train_initial_hidden_state
            if self.train_initial_hidden_state:
                self.hs_initializer, self.lstm_init_h, self.lstm_init_c = self._init_initial_hidden_state(num_directions)

        else:
            self.linear = torch.nn.Linear(embedding_dim, embedding_dim)
            hidden_output_dim = embedding_dim

        # ----- CRF / Linear layer -----
        if use_crf:
            self.crf = CRF(hidden_output_dim, self.tagset_size, init_from_state_dict)
            self.loss_function = ViterbiLoss(tag_dictionary)
            self.viterbi_decoder = ViterbiDecoder(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_output_dim, self.tagset_size)
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")

        self.to(flair.device)

    @property
    def label_type(self):
        return self.tag_type

    def _init_loss_weights(self, loss_weights) -> torch.tensor:
        n_classes = len(self.label_dictionary)
        weight_list = [1. for i in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        return torch.FloatTensor(weight_list).to(flair.device)

    def _init_initial_hidden_state(self, num_directions: int):
        hs_initializer = torch.nn.init.xavier_normal_
        lstm_init_h = torch.nn.Parameter(
            torch.randn(self.nlayers * num_directions, self.hidden_size),
            requires_grad=True,
        )
        lstm_init_c = torch.nn.Parameter(
            torch.randn(self.nlayers * num_directions, self.hidden_size),
            requires_grad=True,
        )

        return hs_initializer, lstm_init_h, lstm_init_c

    @staticmethod
    def RNN(
            rnn_type: str,
            rnn_layers: int,
            hidden_size: int,
            bidirectional: bool,
            rnn_input_dim: int
    ) -> torch.nn.Module:
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

    def forward_pass(self,
                     sentences: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ) -> tuple:

        self.embeddings.embed(sentences)

        # Get embedding for each sentence
        tensor_list, lengths = self._create_tensor_list(sentences)
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)
        lengths = lengths.sort(dim=0, descending=True)

        # sort tensor in decreasing order based on lengths of sentences in batch
        sentences = [sentences[i] for i in lengths.indices]
        sentence_tensor = sentence_tensor[lengths.indices]

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
            packed = pack_padded_sequence(sentence_tensor, lengths.values, batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            sentence_tensor = self.linear(sentence_tensor)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # Depending on whether we are using CRF or a linear layer, scores is either:
        # -- A tensor of shape (batch size, sequence length, tagset size, tagset size) for CRF
        # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
        if self.use_crf:
            features = self.crf(sentence_tensor)
            scores = (features, lengths)
        else:
            features = self.linear2tag(sentence_tensor)
            scores = self._get_scores_from_features(features, lengths)

        gold_labels = self._get_gold_labels(sentences)

        return scores, gold_labels

    def _create_tensor_list(self, sentences: Union[List[DataPoint], DataPoint]) -> tuple:
        tensor_list = list(map(lambda sent: sent.get_sequence_tensor(), sentences))
        lengths = torch.LongTensor([len(sentence) for sentence in sentences])
        return tensor_list, lengths

    @staticmethod
    def _get_scores_from_features(features: torch.tensor, lengths: torch.tensor):
        features_formatted = []
        for feat, length in zip(features, lengths.values):
            features_formatted.append(feat[:length])
        scores = torch.cat(features_formatted)
        return scores

    def _get_gold_labels(self, sentences: Union[List[DataPoint], DataPoint]):
        tokens_per_sentence = [[token for token in sentence] for sentence in sentences]
        labels = [[[token.get_tag(self.label_type).value] for token in sentence] for sentence in tokens_per_sentence]
        labels = [token for sentence in labels for token in sentence]
        return labels

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none"
    ):
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

        if isinstance(sentences, DataPoint):
            sentences = [sentences]

        # filter empty sentences
        if isinstance(sentences[0], DataPoint):
            sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        dataloader = DataLoader(dataset=SentenceDataset(sentences), batch_size=mini_batch_size)
        # progress bar for verbosity
        if verbose:
            dataloader = tqdm(dataloader)

        overall_loss = 0
        batch_no = 0
        label_count = 0
        for batch in dataloader:

            batch_no += 1

            if verbose:
                dataloader.set_description(f"Inferencing on batch {batch_no}")

            # stop if all sentences are empty
            if not batch:
                continue

            features, gold_labels = self.forward_pass(batch)

            # remove previously predicted labels of this type
            for sentence in batch:
                sentence.remove_labels(label_name)

            if return_loss:
                loss = self._calculate_loss(features, gold_labels)
                overall_loss += loss[0]
                label_count += loss[1]

            batch = sorted(batch, key=len, reverse=True)

            if self.use_crf:
                predictions = self.viterbi_decoder.decode(features, transitions=self.crf.transitions)
            else:
                predictions = self._standard_inference(features, batch)

            for sentence, labels in zip(batch, predictions):
                for token, label in zip(sentence.tokens, labels):
                    token.add_tag_label(label_name, label)

        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return overall_loss, label_count

    def _standard_inference(self, features: torch.tensor, batch):
        softmax_batch = F.softmax(features, dim=1).cpu()
        scores_batch, prediction_batch = torch.max(softmax_batch, dim=1)
        predictions = []
        for sentence in batch:
            scores = scores_batch[:len(sentence)]
            predictions_for_sentence = prediction_batch[:len(sentence)]
            predictions.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(prediction), score.item())
                    for token, score, prediction in zip(sentence, scores, predictions_for_sentence)
                ]
            )
            scores_batch = scores_batch[len(sentence):]
            prediction_batch = prediction_batch[len(sentence):]

        return predictions

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "reproject_embeddings": self.reproject_embeddings,
            "weight_dict": self.weight_dict
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
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
            if "linear.weight" in state["state_dict"] and "linear.bias" in state["state_dict"]:
                state["state_dict"]["crf.emission.weight"] = state["state_dict"]["linear.weight"]
                state["state_dict"]["crf.emission.bias"] = state["state_dict"]["linear.bias"]
                del state["state_dict"]["linear.weight"]
                del state["state_dict"]["linear.bias"]

        model = SequenceTagger(
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
            init_from_state_dict=True
        )

        model.load_state_dict(state["state_dict"])
        return model

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

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences


class MultiTagger:
    def __init__(self, name_to_tagger: Dict[str, SequenceTagger]):
        super().__init__()
        self.name_to_tagger = name_to_tagger

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
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
