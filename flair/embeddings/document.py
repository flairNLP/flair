import logging
from typing import Any, Optional, Union, cast

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import RNNBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import flair
from flair.data import Sentence
from flair.embeddings.base import (
    DocumentEmbeddings,
    load_embeddings,
    register_embeddings,
)
from flair.embeddings.token import FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.embeddings.transformer import (
    TransformerEmbeddings,
    TransformerOnnxDocumentEmbeddings,
)
from flair.nn import LockedDropout, WordDropout

log = logging.getLogger("flair")


@register_embeddings
class TransformerDocumentEmbeddings(DocumentEmbeddings, TransformerEmbeddings):
    onnx_cls = TransformerOnnxDocumentEmbeddings

    def __init__(
        self,
        model: str = "bert-base-uncased",  # set parameters with different default values
        layers: str = "-1",
        layer_mean: bool = False,
        is_token_embedding: bool = False,
        **kwargs,
    ) -> None:
        """Bidirectional transformer embeddings of words from various transformer architectures.

        Args:
            model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for options)
            layers: string indicating which layers to take for embedding (-1 is topmost layer)
            cls_pooling: Pooling strategy for combining token level embeddings. options are 'cls', 'max', 'mean'.
            layer_mean: If True, uses a scalar mix of layers as embedding
            fine_tune: If True, allows transformers to be fine-tuned during training
            is_token_embedding: If True, the embedding can be used as TokenEmbedding too.
            **kwargs: Arguments propagated to :meth:`flair.embeddings.transformer.TransformerEmbeddings.__init__`
        """
        TransformerEmbeddings.__init__(
            self,
            model=model,
            layers=layers,
            layer_mean=layer_mean,
            is_token_embedding=is_token_embedding,
            is_document_embedding=True,
            **kwargs,
        )

    @classmethod
    def create_from_state(cls, **state):
        # this parameter is fixed
        del state["is_document_embedding"]
        return cls(**state)


@register_embeddings
class DocumentPoolEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: Union[TokenEmbeddings, list[TokenEmbeddings]],
        fine_tune_mode: str = "none",
        pooling: str = "mean",
    ) -> None:
        """The constructor takes a list of embeddings to be combined.

        Args:
            embeddings: a list of token embeddings
            fine_tune_mode: if set to "linear" a trainable layer is added, if set to "nonlinear", a nonlinearity is added as well. Set this to make the pooling trainable.
            pooling: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()

        if isinstance(embeddings, StackedEmbeddings):
            embeddings = embeddings.embeddings
        elif isinstance(embeddings, TokenEmbeddings):
            embeddings = [embeddings]

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length

        # optional fine-tuning on top of embedding layer
        self.fine_tune_mode = fine_tune_mode
        if self.fine_tune_mode in ["nonlinear", "linear"]:
            self.embedding_flex = torch.nn.Linear(self.embedding_length, self.embedding_length, bias=False)
            self.embedding_flex.weight.data.copy_(torch.eye(self.embedding_length))

        if self.fine_tune_mode in ["nonlinear"]:
            self.embedding_flex_nonlinear = torch.nn.ReLU()
            self.embedding_flex_nonlinear_map = torch.nn.Linear(self.embedding_length, self.embedding_length)

        self.__embedding_length = self.embeddings.embedding_length

        self.to(flair.device)

        if pooling not in ["min", "max", "mean"]:
            raise ValueError(f"Pooling operation for {self.mode!r} is not defined")

        self.pooling = pooling
        self.name: str = f"document_{self.pooling}"

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[list[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences.

        If embeddings are already added, updates only if embeddings are non-static.
        """
        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        for sentence in sentences:
            word_embeddings = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence.tokens], dim=0).to(
                flair.device
            )

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

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        pass

    def extra_repr(self):
        return f"fine_tune_mode={self.fine_tune_mode}, pooling={self.pooling}"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DocumentPoolEmbeddings":
        embeddings = cast(StackedEmbeddings, load_embeddings(params.pop("embeddings"))).embeddings
        return cls(embeddings=embeddings, **params)

    def to_params(self) -> dict[str, Any]:
        return {
            "pooling": self.pooling,
            "fine_tune_mode": self.fine_tune_mode,
            "embeddings": self.embeddings.save_embeddings(False),
        }


@register_embeddings
class DocumentTFIDFEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        train_dataset: list[Sentence],
        vectorizer: Optional[TfidfVectorizer] = None,
        **vectorizer_params,
    ) -> None:
        """The constructor for DocumentTFIDFEmbeddings.

        Args:
            train_dataset: the train dataset which will be used to construct a vectorizer
            vectorizer: a precalculated vectorizer. If provided, requires the train_dataset to be an empty list.
            vectorizer_params: parameters given to Scikit-learn's TfidfVectorizer constructor
        """
        super().__init__()

        import numpy as np

        if vectorizer is not None:
            self.vectorizer = vectorizer
            if len(train_dataset) > 0:
                raise ValueError("Cannot initialize document tfidf embeddings with a vectorizer and with a dataset")
        else:
            self.vectorizer = TfidfVectorizer(dtype=np.float32, **vectorizer_params)
            self.vectorizer.fit([s.to_original_text() for s in train_dataset])

        self.__embedding_length: int = len(self.vectorizer.vocabulary_)

        self.to(flair.device)

        self.name: str = "document_tfidf"
        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[list[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences."""
        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        raw_sentences = [s.to_original_text() for s in sentences]
        tfidf_vectors = torch.from_numpy(self.vectorizer.transform(raw_sentences).A)

        for sentence_id, sentence in enumerate(sentences):
            sentence.set_embedding(self.name, tfidf_vectors[sentence_id])

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        pass

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DocumentTFIDFEmbeddings":
        return cls(train_dataset=[], vectorizer=params["vectorizer"])

    def to_params(self) -> dict[str, Any]:
        return {
            "vectorizer": self.vectorizer,
        }


@register_embeddings
class DocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: list[TokenEmbeddings],
        hidden_size=128,
        rnn_layers=1,
        reproject_words: bool = True,
        reproject_words_dimension: Optional[int] = None,
        bidirectional: bool = False,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        rnn_type: str = "GRU",
        fine_tune: bool = True,
    ) -> None:
        """Instantiates an RNN that works upon some token embeddings.

        Args:
            embeddings: a list of token embeddings
            hidden_size: the number of hidden states in the rnn
            rnn_layers: the number of layers for the rnn
            reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear layer before putting them into the rnn or not
            reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output dimension as before will be taken.
            bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
            dropout: the dropout value to be used
            word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
            locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
            rnn_type: 'GRU' or 'LSTM'
            fine_tune: if True, allow to finetune the embeddings.
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = not fine_tune

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings, self.embeddings_dimension)

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn: RNNBase = torch.nn.LSTM(
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
        self.locked_dropout = LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        """Add embeddings to all sentences in the given list of sentences.

        If embeddings are already added, update only if embeddings are non-static.
        """
        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        self.rnn.zero_grad()
        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: list[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=self.rnn.all_weights[0][0].dtype,
            device=flair.device,
        )

        all_embs: list[torch.Tensor] = []
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
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
        packed = pack_padded_sequence(sentence_tensor, lengths, enforce_sorted=False, batch_first=True)
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

                num_direction = 2 if child_module.__dict__["bidirectional"] else 1
                for layer in range(child_module.__dict__["num_layers"]):
                    for direction in range(num_direction):
                        suffix = "_reverse" if direction == 1 else ""
                        param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                        if child_module.__dict__["bias"]:
                            param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                        param_names = [x.format(layer, suffix) for x in param_names]
                        _flat_weights_names.extend(param_names)

                child_module._flat_weights_names = _flat_weights_names

            child_module._apply(fn)

    def to_params(self):
        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "embeddings": self.embeddings.save_embeddings(False),
            "hidden_size": self.rnn.hidden_size,
            "rnn_layers": self.rnn.num_layers,
            "reproject_words": self.reproject_words,
            "reproject_words_dimension": self.embeddings_dimension,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout.p if self.dropout is not None else 0.0,
            "word_dropout": self.word_dropout.p if self.word_dropout is not None else 0.0,
            "locked_dropout": self.locked_dropout.p if self.locked_dropout is not None else 0.0,
            "rnn_type": self.rnn_type,
            "fine_tune": not self.static_embeddings,
        }

        return model_state

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DocumentRNNEmbeddings":
        stacked_embeddings = load_embeddings(params["embeddings"])
        assert isinstance(stacked_embeddings, StackedEmbeddings)
        return cls(
            embeddings=stacked_embeddings.embeddings,
            hidden_size=params["hidden_size"],
            rnn_layers=params["rnn_layers"],
            reproject_words=params["reproject_words"],
            reproject_words_dimension=params["reproject_words_dimension"],
            bidirectional=params["bidirectional"],
            dropout=params["dropout"],
            word_dropout=params["word_dropout"],
            locked_dropout=params["locked_dropout"],
            rnn_type=params["rnn_type"],
            fine_tune=params["fine_tune"],
        )

    def __setstate__(self, d):
        # re-initialize language model with constructor arguments
        language_model = DocumentRNNEmbeddings(
            embeddings=d["embeddings"],
            hidden_size=d["hidden_size"],
            rnn_layers=d["rnn_layers"],
            reproject_words=d["reproject_words"],
            reproject_words_dimension=d["reproject_words_dimension"],
            bidirectional=d["bidirectional"],
            dropout=d["dropout"],
            word_dropout=d["word_dropout"],
            locked_dropout=d["locked_dropout"],
            rnn_type=d["rnn_type"],
            fine_tune=d["fine_tune"],
        )

        # special handling for deserializing language models
        if "state_dict" in d:
            language_model.load_state_dict(d["state_dict"])

        # copy over state dictionary to self
        for key in language_model.__dict__:
            self.__dict__[key] = language_model.__dict__[key]

        # set the language model to eval() by default (this is necessary since FlairEmbeddings "protect" the LM
        # in their "self.train()" method)
        self.eval()


@register_embeddings
class DocumentLMEmbeddings(DocumentEmbeddings):
    def __init__(self, flair_embeddings: list[FlairEmbeddings]) -> None:
        super().__init__()

        self.embeddings = flair_embeddings
        self.name = "document_lm"

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(flair_embeddings):
            self.add_module(f"lm_embedding_{i}", embedding)
            if not embedding.static_embeddings:
                self.static_embeddings = False

        self._embedding_length: int = sum(embedding.embedding_length for embedding in flair_embeddings)
        self.eval()

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        for embedding in self.embeddings:
            embedding.embed(sentences)

            # iterate over sentences
            for sentence in sentences:
                # if its a forward LM, take last state
                if embedding.is_forward_lm:
                    sentence.set_embedding(
                        embedding.name,
                        sentence[len(sentence) - 1]._embeddings[embedding.name],
                    )
                else:
                    sentence.set_embedding(embedding.name, sentence[0]._embeddings[embedding.name])

        return sentences

    def get_names(self) -> list[str]:
        if "__names" not in self.__dict__:
            self.__names = [name for embedding in self.embeddings for name in embedding.get_names()]

        return self.__names

    def to_params(self) -> dict[str, Any]:
        return {"flair_embeddings": [embedding.save_embeddings(False) for embedding in self.embeddings]}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DocumentLMEmbeddings":
        return cls([cast(FlairEmbeddings, load_embeddings(embedding)) for embedding in params["flair_embeddings"]])


@register_embeddings
class SentenceTransformerDocumentEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        model: str = "bert-base-nli-mean-tokens",
        batch_size: int = 1,
    ) -> None:
        """Instantiates a document embedding using the SentenceTransformer Embeddings.

        Args:
            model: string name of models from SentencesTransformer Class
            batch_size: int number of sentences to processed in one batch
        """
        super().__init__()

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "sentence-transformers" is not installed!')
            log.warning('To use Sentence Transformers, please first install with "pip install sentence-transformers"')
            log.warning("-" * 100)

        self.model_name = model
        self.model = SentenceTransformer(
            model, cache_folder=str(flair.cache_root / "embeddings" / "sentence-transformer")
        )
        self.name = "sentence-transformers-" + str(model)
        self.batch_size = batch_size
        self.static_embeddings = True
        self.eval()

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        sentence_batches = [
            sentences[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)
        ]

        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    def _add_embeddings_to_sentences(self, sentences: list[Sentence]):
        # convert to plain strings, embedded in a list for the encode function
        sentences_plain_text = [sentence.to_plain_string() for sentence in sentences]

        embeddings = self.model.encode(sentences_plain_text, convert_to_numpy=False)
        for sentence, embedding in zip(sentences, embeddings):
            sentence.set_embedding(self.name, embedding)

    @property
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.model.get_sentence_embedding_dimension()

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "SentenceTransformerDocumentEmbeddings":
        return cls(**params)

    def to_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "batch_size": self.batch_size,
        }


@register_embeddings
class DocumentCNNEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: list[TokenEmbeddings],
        kernels=((100, 3), (100, 4), (100, 5)),
        reproject_words: bool = True,
        reproject_words_dimension: Optional[int] = None,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        fine_tune: bool = True,
    ) -> None:
        """Instantiates a CNN that works upon some token embeddings.

        Args:
            embeddings: a list of token embeddings
            kernels: list of (number of kernels, kernel size)
            reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear layer before putting them into the rnn or not
            reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output dimension as before will be taken.
            dropout: the dropout value to be used
            word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
            locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
            fine_tune: if True, allow to finetune the embeddings.
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.kernels = kernels
        self.reproject_words = reproject_words

        self.static_embeddings = not fine_tune

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        if self.reproject_words:
            self.word_reprojection_map: Optional[torch.nn.Linear] = torch.nn.Linear(
                self.length_of_all_token_embeddings, self.embeddings_dimension
            )
            torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
        else:
            self.word_reprojection_map = None

        # CNN
        self.__embedding_length: int = sum([kernel_num for kernel_num, kernel_size in self.kernels])
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(self.embeddings_dimension, kernel_num, kernel_size)
                for kernel_num, kernel_size in self.kernels
            ]
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

        self.name = "document_cnn"

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        self.to(flair.device)
        self.min_sequence_length = max(kernel_size for _, kernel_size in self.kernels)
        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        """Add embeddings to all sentences in the given list of sentences.

        If embeddings are already added, update only if embeddings are non-static.
        """
        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        self.zero_grad()  # is it necessary?

        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: list[int] = [len(sentence.tokens) for sentence in sentences]
        padding_length: int = max(max(lengths), self.min_sequence_length)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * padding_length,
            dtype=self.convs[0].weight.dtype,
            device=flair.device,
        )

        all_embs: list[torch.Tensor] = []
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = padding_length - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                padding_length,
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
        if self.word_reprojection_map is not None:
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
        for sentence_no, _length in enumerate(lengths):
            embedding = outputs[sentence_no]

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):
        for child_module in self.children():
            child_module._apply(fn)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DocumentCNNEmbeddings":
        embeddings = cast(StackedEmbeddings, load_embeddings(params.pop("embeddings"))).embeddings
        return cls(embeddings=embeddings, **params)

    def to_params(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings.save_embeddings(False),
            "kernels": self.kernels,
            "reproject_words": self.reproject_words,
            "reproject_words_dimension": self.embeddings_dimension,
            "dropout": 0.0 if self.dropout is None else self.dropout.p,
            "word_dropout": 0.0 if self.word_dropout is None else self.word_dropout.p,
            "locked_dropout": 0.0 if self.locked_dropout is None else self.locked_dropout.dropout_rate,
            "fine_tune": not self.static_embeddings,
        }
