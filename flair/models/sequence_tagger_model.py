import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Union, List

# Torch Imports
import torch
import torch.nn
import torch.nn.functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# Flair imports
import flair.nn
from flair.data import Sentence, Dictionary
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.training_utils import Result, store_embeddings
from flair.file_utils import cached_path

# Sequence tagger utils imports
from .sequence_tagger_utils.crf import CRF
from .sequence_tagger_utils.viterbi import ViterbiLoss, ViterbiDecoder
from .sequence_tagger_utils.utils import init_stop_tag_embedding, get_tags_tensor, START_TAG, STOP_TAG

log = logging.getLogger("flair")


class SequenceTagger(flair.nn.Model):

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
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
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
        :param beta: Beta value for evaluation metric.
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        """
        super(SequenceTagger, self).__init__()

        # ----- Embedding specific parameters -----
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length
        self.stop_token_emb = init_stop_tag_embedding(embedding_dim)
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.tag_type = tag_type

        # ----- Evaluation metric parameters -----
        # TODO change metrics evaluation
        #self.metric = Metric("Evaluation", beta=beta)
        self.beta = beta

        # ----- Initial loss weights parameters -----
        self.weight_dict = loss_weights
        self.loss_weights = self.init_loss_weights(loss_weights) if loss_weights else None

        # ----- RNN specific parameters -----
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type if not rnn else rnn._get_name()
        self.hidden_size = hidden_size if not rnn else rnn.hidden_size
        self.rnn_layers = rnn_layers if not rnn else rnn.num_layers
        self.bidirectional = bidirectional if not rnn else rnn.bidirectional

        # ----- Conditional Random Field parameters -----
        self.use_crf = use_crf
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # ----- Dropout parameters -----
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout = True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # ----- Model layers -----
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        if use_rnn:
            # If Shared RNN provided, create one for model
            if not rnn:
                self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim=embedding_dim)
            else:
                self.rnn = rnn
            num_directions = 2 if self.bidirectional else 1
            hidden_output_dim = self.rnn.hidden_size * num_directions
        else:
            self.linear = torch.nn.Linear(embedding_dim, embedding_dim)
            hidden_output_dim = embedding_dim

        if use_crf:
            self.crf = CRF(hidden_output_dim, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary)
            self.viterbi_decoder = ViterbiDecoder(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_output_dim, self.tagset_size)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    @property
    def label_type(self):
        return self.tag_type

    def init_loss_weights(self, loss_weights) -> torch.Tensor:
        """
        Initialize loss weights.
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1. for i in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        return torch.FloatTensor(weight_list).to(flair.device)

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

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

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward loss function from abstract base class flair.nn.Model
        :param sentences: batch of sentences
        """
        features, lengths = self.forward(sentences)
        return self.loss(features, sentences, lengths)

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> (torch.Tensor, torch.Tensor):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        :return: features and lengths of each sentence in batch
        """
        self.embeddings.embed(sentences)

        # Get embedding for each sentence + append a stop token embedding to each sentence
        tensor_list = list(map(lambda sent: torch.cat((sent.get_sequence_tensor(), self.stop_token_emb.unsqueeze(0)), dim=0), sentences))
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        # +1 since we've added a stop token embedding to each sentence
        lengths = torch.LongTensor([len(sentence) + 1 for sentence in sentences])
        lengths = lengths.sort(dim=0, descending=True)
        # sort tensor in decreasing order based on lengths of sentences in batch
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
            packed = pack_padded_sequence(sentence_tensor, list(lengths.values), batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            sentence_tensor = self.linear(sentence_tensor)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features, lengths

    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence], lengths) -> torch.Tensor:
        """
        Loss function of multitask base model.
        :param features: Output features / CRF scores from feed-forward function
        :param sentences: batch of sentences
        :param lengths: lenghts of sentences in batch to sort tag tensor accordingly
        """
        tags_tensor = get_tags_tensor(sentences, self.tag_dictionary, self.tag_type)
        tags_tensor = tags_tensor[lengths.indices]

        if self.use_crf:
            loss = self.viterbi_loss(features, tags_tensor, lengths.values)
        else:
            loss = self.cross_entropy_loss(features.permute(0,2,1), tags_tensor)

        return loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            label_name: Optional[str] = None,
            return_loss: bool = False,
            embedding_storage_mode: str ="none",
    ) -> Optional[torch.Tensor]:
        """
        Predicting tag sequence for current batch of sentences.
        :param sentences: batch of sentences
        :param label_name: which label should be predicted
        :param return_loss: If True, a loss float tensor is returned
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
            freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Can return a loss float value (Tensor)
        """
        sentences = self._filter_empty_sentences(sentences)

        if label_name == None:
            label_name = self.tag_type

        features, lengths = self.forward(sentences)

        # features und lengths in der forward sortiert
        tags = self.viterbi_decoder.decode(features, lengths)

        # sorted sentences to match tags from decoder
        sentences = [sentences[i] for i in lengths.indices]

        # Add predicted labels to sentences
        for (sentence, sent_tags) in zip(sentences, tags):
            for (token, tag) in zip(sentence.tokens, sent_tags):
                token.add_tag_label(label_name, tag)

        # clearing token embeddings to save memory
        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return self.loss(features, sentences, lengths)

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
            "beta": self.beta,
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
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]

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
            rnn_type=rnn_type,
            beta=beta,
            reproject_embeddings=reproject_embeddings,
            loss_weights=weights
        )
        model.load_state_dict(state["state_dict"])
        return model


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
