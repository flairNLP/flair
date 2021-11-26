import logging
from abc import abstractmethod
from typing import List, Union, Dict

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoConfig, AutoModel, CONFIG_MAPPING, PreTrainedTokenizer

import flair
from flair.data import Sentence
from flair.embeddings.base import Embeddings, ScalarMix
from flair.embeddings.token import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.nn import LockedDropout, WordDropout

log = logging.getLogger("flair")


class DocumentEmbeddings(Embeddings):
    """Abstract base class for all document-level embeddings. Every new type of document embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "sentence-level"


class TransformerDocumentEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            model: str = "bert-base-uncased",
            fine_tune: bool = True,
            layers: str = "-1",
            layer_mean: bool = False,
            pooling: str = "cls",
            **kwargs
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param layer_mean: If True, uses a scalar mix of layers as embedding
        :param pooling: Pooling strategy for combining token level embeddings. options are 'cls', 'max', 'mean'.
        """
        super().__init__()

        if pooling not in ['cls', 'max', 'mean']:
            raise ValueError(f"Pooling operation `{pooling}` is not defined for TransformerDocumentEmbeddings")

        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # do not print transformer warnings as these are confusing in this case
        from transformers import logging
        logging.set_verbosity_error()

        # load tokenizer and transformer model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        if self.tokenizer.model_max_length > 1000000000:
            self.tokenizer.model_max_length = 512
            log.info("No model_max_length in Tokenizer's config.json - setting it to 512. "
                     "Specify desired model_max_length by passing it as attribute to embedding instance.")
        if not 'config' in kwargs:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
            self.model = AutoModel.from_pretrained(model, config=config)
        else:
            self.model = AutoModel.from_pretrained(None, **kwargs)

        logging.set_verbosity_warning()

        # model name
        self.name = 'transformer-document-' + str(model)
        self.base_model_name = str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(flair.device)

        # embedding parameters
        if layers == 'all':
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]

        self.layer_mean = layer_mean
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.pooling = pooling

        # check whether CLS is at beginning or end
        self.initial_cls_token: bool = self._has_initial_cls_token(tokenizer=self.tokenizer)

    @staticmethod
    def _has_initial_cls_token(tokenizer: PreTrainedTokenizer) -> bool:
        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        tokens = tokenizer.encode('a')
        initial_cls_token: bool = False
        if tokens[0] == tokenizer.cls_token_id: initial_cls_token = True
        return initial_cls_token

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences."""

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # first, subtokenize each sentence and find out into how many subtokens each token was divided
            subtokenized_sentences = []

            # subtokenize sentences
            for sentence in sentences:
                # tokenize and truncate to max subtokens (TODO: check better truncation strategies)
                subtokenized_sentence = self.tokenizer.encode(sentence.to_tokenized_string(),
                                                              add_special_tokens=True,
                                                              max_length=self.tokenizer.model_max_length,
                                                              truncation=True,
                                                              )

                subtokenized_sentences.append(
                    torch.tensor(subtokenized_sentence, dtype=torch.long, device=flair.device))

            # find longest sentence in batch
            longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

            # initialize batch tensors and mask
            input_ids = torch.zeros(
                [len(sentences), longest_sequence_in_batch],
                dtype=torch.long,
                device=flair.device,
            )
            mask = torch.zeros(
                [len(sentences), longest_sequence_in_batch],
                dtype=torch.long,
                device=flair.device,
            )
            for s_id, sentence in enumerate(subtokenized_sentences):
                sequence_length = len(sentence)
                input_ids[s_id][:sequence_length] = sentence
                mask[s_id][:sequence_length] = torch.ones(sequence_length)

            # put encoded batch through transformer model to get all hidden states of all encoder layers
            hidden_states = self.model(input_ids, attention_mask=mask)[-1] if len(sentences) > 1 \
                else self.model(input_ids)[-1]

            # iterate over all subtokenized sentences
            for sentence_idx, (sentence, subtokens) in enumerate(zip(sentences, subtokenized_sentences)):

                if self.pooling == "cls":
                    index_of_CLS_token = 0 if self.initial_cls_token else len(subtokens) - 1

                    cls_embeddings_all_layers: List[torch.FloatTensor] = \
                        [hidden_states[layer][sentence_idx][index_of_CLS_token] for layer in self.layer_indexes]

                    embeddings_all_layers = cls_embeddings_all_layers

                elif self.pooling == "mean":
                    mean_embeddings_all_layers: List[torch.FloatTensor] = \
                        [torch.mean(hidden_states[layer][sentence_idx][:len(subtokens), :], dim=0) for layer in
                         self.layer_indexes]

                    embeddings_all_layers = mean_embeddings_all_layers

                elif self.pooling == "max":
                    max_embeddings_all_layers: List[torch.FloatTensor] = \
                        [torch.max(hidden_states[layer][sentence_idx][:len(subtokens), :], dim=0)[0] for layer in
                         self.layer_indexes]

                    embeddings_all_layers = max_embeddings_all_layers

                # use scalar mix of embeddings if so selected
                if self.layer_mean:
                    sm = ScalarMix(mixture_size=len(embeddings_all_layers))
                    sm_embeddings = sm(embeddings_all_layers)

                    embeddings_all_layers = [sm_embeddings]

                # set the extracted embedding for the token
                sentence.set_embedding(self.name, torch.cat(embeddings_all_layers))

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.layer_mean
            else self.model.config.hidden_size
        )

    def __getstate__(self):
        # special handling for serializing transformer models
        config_state_dict = self.model.config.__dict__
        model_state_dict = self.model.state_dict()

        if not hasattr(self, "base_model_name"): self.base_model_name = self.name.split('transformer-document-')[-1]

        # serialize the transformer models and the constructor arguments (but nothing else)
        model_state = {
            "config_state_dict": config_state_dict,
            "model_state_dict": model_state_dict,
            "embedding_length_internal": self.embedding_length,

            "base_model_name": self.base_model_name,
            "fine_tune": self.fine_tune,
            "layer_indexes": self.layer_indexes,
            "layer_mean": self.layer_mean,
            "pooling": self.pooling,
        }

        return model_state

    def __setstate__(self, d):
        self.__dict__ = d

        # necessary for reverse compatibility with Flair <= 0.7
        if 'use_scalar_mix' in self.__dict__.keys():
            self.__dict__['layer_mean'] = d['use_scalar_mix']

        # special handling for deserializing transformer models
        if "config_state_dict" in d:

            # load transformer model
            model_type = d["config_state_dict"]["model_type"] if "model_type" in d["config_state_dict"] else "bert"
            config_class = CONFIG_MAPPING[model_type]
            loaded_config = config_class.from_dict(d["config_state_dict"])

            # constructor arguments
            layers = ','.join([str(idx) for idx in self.__dict__['layer_indexes']])

            # re-initialize transformer word embeddings with constructor arguments
            embedding = TransformerDocumentEmbeddings(
                model=self.__dict__['base_model_name'],
                fine_tune=self.__dict__['fine_tune'],
                layers=layers,
                layer_mean=self.__dict__['layer_mean'],

                config=loaded_config,
                state_dict=d["model_state_dict"],
                pooling=self.__dict__['pooling'] if 'pooling' in self.__dict__ else 'cls',
                # for backward compatibility with previous models
            )

            # I have no idea why this is necessary, but otherwise it doesn't work
            for key in embedding.__dict__.keys():
                self.__dict__[key] = embedding.__dict__[key]

        else:
            model_name = self.__dict__['name'].split('transformer-document-')[-1]
            # reload tokenizer to get around serialization issues
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                pass
            self.tokenizer = tokenizer


class DocumentPoolEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            fine_tune_mode: str = "none",
            pooling: str = "mean",
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param fine_tune_mode: if set to "linear" a trainable layer is added, if set to
        "nonlinear", a nonlinearity is added as well. Set this to make the pooling trainable.
        :param pooling: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length

        # optional fine-tuning on top of embedding layer
        self.fine_tune_mode = fine_tune_mode
        if self.fine_tune_mode in ["nonlinear", "linear"]:
            self.embedding_flex = torch.nn.Linear(
                self.embedding_length, self.embedding_length, bias=False
            )
            self.embedding_flex.weight.data.copy_(torch.eye(self.embedding_length))

        if self.fine_tune_mode in ["nonlinear"]:
            self.embedding_flex_nonlinear = torch.nn.ReLU(self.embedding_length)
            self.embedding_flex_nonlinear_map = torch.nn.Linear(
                self.embedding_length, self.embedding_length
            )

        self.__embedding_length: int = self.embeddings.embedding_length

        self.to(flair.device)

        if pooling not in ['min', 'max', 'mean']:
            raise ValueError(f"Pooling operation for {self.mode!r} is not defined")

        self.pooling = pooling
        self.name: str = f"document_{self.pooling}"

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        for sentence in sentences:
            word_embeddings = []
            for token in sentence.tokens:
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)

            if self.fine_tune_mode in ["nonlinear", "linear"]:
                word_embeddings = self.embedding_flex(word_embeddings)

            if self.fine_tune_mode in ["nonlinear"]:
                word_embeddings = self.embedding_flex_nonlinear(word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(word_embeddings)

            if self.pooling == "mean":
                pooled_embedding = torch.mean(word_embeddings, 0)
            elif self.pooling == "max":
                pooled_embedding, _ = torch.max(word_embeddings, 0)
            elif self.pooling == "min":
                pooled_embedding, _ = torch.min(word_embeddings, 0)

            sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass

    def extra_repr(self):
        return f"fine_tune_mode={self.fine_tune_mode}, pooling={self.pooling}"


class DocumentTFIDFEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            train_dataset,
            **vectorizer_params,
    ):
        """The constructor for DocumentTFIDFEmbeddings.
        :param train_dataset: the train dataset which will be used to construct vectorizer
        :param vectorizer_params: parameters given to Scikit-learn's TfidfVectorizer constructor
        """
        super().__init__()

        import numpy as np
        self.vectorizer = TfidfVectorizer(dtype=np.float32, **vectorizer_params)
        self.vectorizer.fit([s.to_original_text() for s in train_dataset])

        self.__embedding_length: int = len(self.vectorizer.vocabulary_)

        self.to(flair.device)

        self.name: str = f"document_tfidf"

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences."""

        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        raw_sentences = [s.to_original_text() for s in sentences]
        tfidf_vectors = torch.from_numpy(self.vectorizer.transform(raw_sentences).A)

        for sentence_id, sentence in enumerate(sentences):
            sentence.set_embedding(self.name, tfidf_vectors[sentence_id])

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            hidden_size=128,
            rnn_layers=1,
            reproject_words: bool = True,
            reproject_words_dimension: int = None,
            bidirectional: bool = False,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            rnn_type="GRU",
            fine_tune: bool = True,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False if fine_tune else True

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = (
            LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        )
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
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

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push through RNN
        packed = pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        rnn_out, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(rnn_out, batch_first=True)

        # after-RNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract embeddings from RNN
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[sentence_no, length - 1]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[sentence_no, 0]
                embedding = torch.cat([first_rep, last_rep], 0)

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):

        # models that were serialized using torch versions older than 1.4.0 lack the _flat_weights_names attribute
        # check if this is the case and if so, set it
        for child_module in self.children():
            if isinstance(child_module, torch.nn.RNNBase) and not hasattr(child_module, "_flat_weights_names"):
                _flat_weights_names = []

                if child_module.__dict__["bidirectional"]:
                    num_direction = 2
                else:
                    num_direction = 1
                for layer in range(child_module.__dict__["num_layers"]):
                    for direction in range(num_direction):
                        suffix = "_reverse" if direction == 1 else ""
                        param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                        if child_module.__dict__["bias"]:
                            param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                        param_names = [
                            x.format(layer, suffix) for x in param_names
                        ]
                        _flat_weights_names.extend(param_names)

                setattr(child_module, "_flat_weights_names",
                        _flat_weights_names)

            child_module._apply(fn)

    def __getstate__(self):

        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "state_dict": self.state_dict(),

            "embeddings": self.embeddings.embeddings,
            "hidden_size": self.rnn.hidden_size,
            "rnn_layers": self.rnn.num_layers,
            "reproject_words": self.reproject_words,
            "reproject_words_dimension": self.embeddings_dimension,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout.p if self.dropout is not None else 0.,
            "word_dropout": self.word_dropout.p if self.word_dropout is not None else 0.,
            "locked_dropout": self.locked_dropout.p if self.locked_dropout is not None else 0.,
            "rnn_type": self.rnn_type,
            "fine_tune": not self.static_embeddings,
        }

        return model_state

    def __setstate__(self, d):

        # special handling for deserializing language models
        if "state_dict" in d:

            # re-initialize language model with constructor arguments
            language_model = DocumentRNNEmbeddings(
                embeddings=d['embeddings'],
                hidden_size=d['hidden_size'],
                rnn_layers=d['rnn_layers'],
                reproject_words=d['reproject_words'],
                reproject_words_dimension=d['reproject_words_dimension'],
                bidirectional=d['bidirectional'],
                dropout=d['dropout'],
                word_dropout=d['word_dropout'],
                locked_dropout=d['locked_dropout'],
                rnn_type=d['rnn_type'],
                fine_tune=d['fine_tune'],
            )

            language_model.load_state_dict(d['state_dict'])

            # copy over state dictionary to self
            for key in language_model.__dict__.keys():
                self.__dict__[key] = language_model.__dict__[key]

            # set the language model to eval() by default (this is necessary since FlairEmbeddings "protect" the LM
            # in their "self.train()" method)
            self.eval()

        else:
            self.__dict__ = d


class DocumentLMEmbeddings(DocumentEmbeddings):
    def __init__(self, flair_embeddings: List[FlairEmbeddings]):
        super().__init__()

        self.embeddings = flair_embeddings
        self.name = "document_lm"

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(flair_embeddings):
            self.add_module("lm_embedding_{}".format(i), embedding)
            if not embedding.static_embeddings:
                self.static_embeddings = False

        self._embedding_length: int = sum(
            embedding.embedding_length for embedding in flair_embeddings
        )

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

            # iterate over sentences
            for sentence in sentences:
                sentence: Sentence = sentence

                # if its a forward LM, take last state
                if embedding.is_forward_lm:
                    sentence.set_embedding(
                        embedding.name,
                        sentence[len(sentence) - 1]._embeddings[embedding.name],
                    )
                else:
                    sentence.set_embedding(
                        embedding.name, sentence[0]._embeddings[embedding.name]
                    )

        return sentences


class SentenceTransformerDocumentEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            model: str = "bert-base-nli-mean-tokens",
            batch_size: int = 1,
            convert_to_numpy: bool = False,
    ):
        """
        :param model: string name of models from SentencesTransformer Class
        :param name: string name of embedding type which will be set to Sentence object
        :param batch_size: int number of sentences to processed in one batch
        :param convert_to_numpy: bool whether the encode() returns a numpy array or PyTorch tensor
        """
        super().__init__()

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "sentence-transformers" is not installed!')
            log.warning(
                'To use Sentence Transformers, please first install with "pip install sentence-transformers"'
            )
            log.warning("-" * 100)
            pass

        self.model = SentenceTransformer(model)
        self.name = 'sentence-transformers-' + str(model)
        self.batch_size = batch_size
        self.convert_to_numpy = convert_to_numpy
        self.static_embeddings = True

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):

        # convert to plain strings, embedded in a list for the encode function
        sentences_plain_text = [sentence.to_plain_string() for sentence in sentences]

        embeddings = self.model.encode(sentences_plain_text, convert_to_numpy=self.convert_to_numpy)
        for sentence, embedding in zip(sentences, embeddings):
            sentence.set_embedding(self.name, embedding)

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.model.get_sentence_embedding_dimension()


class DocumentCNNEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            kernels=((100, 3), (100, 4), (100, 5)),
            reproject_words: bool = True,
            reproject_words_dimension: int = None,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            fine_tune: bool = True,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param kernels: list of (number of kernels, kernel size)
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.kernels = kernels
        self.reproject_words = reproject_words

        self.static_embeddings = False if fine_tune else True

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # CNN
        self.__embedding_length: int = sum([kernel_num for kernel_num, kernel_size in self.kernels])
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(self.embeddings_dimension, kernel_num, kernel_size) for kernel_num, kernel_size in
                self.kernels
            ]
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

        self.name = "document_cnn"

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = (
            LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        )
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.zero_grad()  # is it necessary?

        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
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

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push CNN
        x = sentence_tensor
        x = x.permute(0, 2, 1)

        rep = [self.pool(torch.nn.functional.relu(conv(x))) for conv in self.convs]
        outputs = torch.cat(rep, 1)

        outputs = outputs.reshape(outputs.size(0), -1)

        # after-CNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract embeddings from CNN
        for sentence_no, length in enumerate(lengths):
            embedding = outputs[sentence_no]

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):
        for child_module in self.children():
            child_module._apply(fn)


class StackedDocumentEmbeddings(DocumentEmbeddings):
    """A stack of document embeddings of different types.
    Multiple sentence level representations are combined into one vector.
    This class accepts any type of document embeddings and different models can
    be trained together (e.g. TransformerDocumentEmbeddings, DocumentPoolEmbeddings)."""

    def __init__(self,
            embeddings: List[DocumentEmbeddings],
            combine_method: str = 'concat',
            alternate_freezing_prob: float = 0.,
            alternate_dropout_prob: float = 0.,
            dropout_prob: float = 0.,
    ):
        """
        The constructor takes a list of document-level embeddings to be combined.
        :param embeddings: list of different document embeddings
        :param combine_method: one of three methods to combine embeddings ["concat", "mean", or "weighted-sum"]
        :param alternate_freezing_prob: probability of one model to be frozen during fine-tuning
        :param alternate_dropout_prob: standard dropout probability to be applied to one of the models
        :param dropout_prob: standard dropout probability on top of the combined embedding
        """
        super().__init__()

        self.embeddings = embeddings
        self.combine_method = combine_method
        self.alternate_freezing_prob = alternate_freezing_prob
        self.alternate_dropout_prob = alternate_dropout_prob
        self.dropout_prob = dropout_prob
        self.num_document_embeddings = len(embeddings)
        self.__use_projection = False

        self.name: str = "Document Embedding Stack"

        # add encoder models as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"document_embedding_{str(i)}", embedding)

        if combine_method not in ["concat", "mean", "weighted-sum"]:
            raise ValueError(f"We only support concatenation, mean pooling, and weighted summation "
                f"('concat', 'mean', 'weighted-sum') as method to combine different embedding types. "
                f"`{combine_method}` is not supported.")

        embedding_lengths = [embedding.embedding_length for embedding in embeddings]

        # do not combine embeddings if only one model is used
        if self.num_document_embeddings == 1:
            self.__embedding_length = embedding_lengths[0]
            self.combine_method = None

        # concatenation of different embedding types
        if self.combine_method == "concat":
            self.__embedding_length = sum(embedding_lengths)

        # add embedding projections only if embeddings are of different sizes
        if self.combine_method != "concat":
            if not all(emb_length == embedding_lengths[0] for emb_length in embedding_lengths):
                self.__use_projection = True

        # project embeddings into the same space
        projection_size = max(embedding_lengths)
        if self.__use_projection:
            self.__embedding_length = projection_size
            for i, embedding in enumerate(embeddings):
                projection = torch.nn.Linear(embedding_lengths[i], projection_size)
                self.add_module(f"embedding_projection_{str(i)}", projection)

        # average multiple embedding types
        if self.combine_method == "mean":
            self.__embedding_length = projection_size

        # weighted summation of different embedding types
        if self.combine_method == 'weighted-sum':
            self.__embedding_length = projection_size
            self.attention = torch.nn.Linear(projection_size, 1)

        # alternate freezing: uniform distribution to freeze one of the encoders
        if alternate_freezing_prob > 0.:
            if not all(isinstance(model, TransformerDocumentEmbeddings) for model in embeddings):
                raise ValueError(f"Alternate freezing is only possible when "
                                 f"combining different transformers.")
            else:
                self.alternate_dist = torch.distributions.uniform.Uniform(0, 1)

        # apply standard dropout to one of the encoders
        if alternate_dropout_prob > 0.:
            self.alternate_dropout = torch.nn.Dropout(alternate_dropout_prob)

        # standard dropout on top of the combined embedding
        # TODO: change to locked dropout
        if self.dropout_prob > 0.:
            self.dropout = torch.nn.Dropout(dropout_prob)


    def embed(
            self,
            sentences: Union[Sentence, List[Sentence]],
            embed_with_attention: bool = False
    ):
        # if only one sentence is passed, convert to list
        if type(sentences) is Sentence:
            sentences = [sentences]

        # random choice from one of the models
        dice = torch.randint(self.num_document_embeddings, (1,)).item()

        # alternate freezing: freeze one of the encoders
        if self.alternate_freezing_prob > 0.:
            if self.alternate_dist.sample() < self.alternate_freezing_prob:
                encoder = getattr(self, f"document_embedding_{str(dice)}")
                encoder.fine_tune = False

        # embed sentence with all encoders
        for embedding in self.embeddings:
            embedding.embed(sentences)

        # stack embeddings of different encoders
        sentence_embeddings = [ [] for _ in range(self.num_document_embeddings) ]
        for sentence in sentences:
            for i, embedding_type in enumerate(list(sentence._embeddings.values())):
                sentence_embeddings[i].append(embedding_type)
        sentence_embeddings = [torch.stack(embedding_type) for embedding_type in sentence_embeddings]

        # alternate freezing: unfreeze all encoders
        if self.alternate_freezing_prob > 0.:
            for i in range(self.num_document_embeddings):
                encoder = getattr(self, f"document_embedding_{i}")
                encoder.fine_tune = True

        # do not combine embeddings if only one model is used
        if not self.combine_method:
            meta_embeddings = sentence_embeddings[0]

        # project embeddings
        if self.__use_projection:
            for i in range(self.num_document_embeddings):
                projection = getattr(self, f"embedding_projection_{i}")
                sentence_embeddings[i] = projection(sentence_embeddings[i])

        # apply dropout to a randomly chosen encoder
        if self.alternate_dropout_prob > 0.:
            sentence_embeddings[dice] = self.alternate_dropout(sentence_embeddings[dice])

        if self.combine_method == "concat":
            meta_embeddings = torch.cat(sentence_embeddings, dim=-1)

        if self.combine_method == "mean":
            sentence_embeddings = torch.stack(sentence_embeddings, dim=1)
            meta_embeddings = torch.mean(sentence_embeddings, dim=1)

        if self.combine_method == "weighted-sum":
            sentence_embeddings = torch.stack(sentence_embeddings, dim=1)
            attn_scores = torch.softmax(self.attention(sentence_embeddings), dim=1)
            meta_embeddings = attn_scores * sentence_embeddings
            meta_embeddings = torch.mean(meta_embeddings, dim=1)

        # apply standard dropout on top of the meta-embedding
        if self.dropout_prob > 0.:
            meta_embeddings = self.dropout(meta_embeddings)

        # set a new sentence embedding called 'meta-embedding'
        # and store this embedding in the embedding dictionary for each sentence instance
        for sentence, meta_embedding in zip(sentences, meta_embeddings):
            sentence.set_embedding("meta-embedding", meta_embedding)

        # can be used for visualization purposes of weighted summation
        if embed_with_attention:
            return attn_scores

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedDocumentEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> List[str]:
        """StackedDocumentEmbeddings adds a new dictionary entry for each sentence called meta-embedding.
        This embedding can be used together with TextClassifier."""

        return ["meta-embedding"]

    def get_named_embeddings_dict(self) -> Dict:

        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict

    def embed_with_attention(self, sentences: Union[Sentence, List[Sentence]]):
        """Just for visualization purposes and testing of weighted summation approach.
        You can embed a few sentences and look at attention scores. These scores show how much
        each embedding type is weighted when embedding a sentence"""

        if self.combine_method != 'weighted-sum':
            raise ValueError(f"You need to choose 'weighted-sum' as combine method and train the model."
                f"Current combine method is {self.combine_method} which does not use attention.")

        if self.num_document_embeddings < 2:
            raise ValueError("You need to train more than one language model"
                "if you want to create meta-embedding")

        return self.embed(sentences, embed_with_attention=True)
