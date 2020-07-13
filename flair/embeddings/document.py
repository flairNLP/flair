from abc import abstractmethod
import logging
from typing import List, Union

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer, AutoConfig, AutoModel

import flair
from flair.data import Sentence
from flair.embeddings.base import Embeddings, ScalarMix
from flair.embeddings.token import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.nn import LockedDropout, WordDropout

log = logging.getLogger("flair")


class DocumentEmbeddings(Embeddings):
    """Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods."""

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
        batch_size: int = 1,
        layers: str = "-1",
        use_scalar_mix: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        """
        super().__init__()

        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model, config=config)

        # model name
        self.name = 'transformer-document-' + str(model)

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

        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size

        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        self.initial_cls_token: bool = False
        if isinstance(self.tokenizer, BertTokenizer) or isinstance(self.tokenizer, AlbertTokenizer):
            self.initial_cls_token = True

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences."""

        # using list comprehension
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Extract sentence embedding from CLS token or similar and add to Sentence object."""

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # first, subtokenize each sentence and find out into how many subtokens each token was divided
            subtokenized_sentences = []

            # subtokenize sentences
            for sentence in sentences:
                # tokenize and truncate to 512 subtokens (TODO: check better truncation strategies)
                subtokenized_sentence = self.tokenizer.encode(sentence.to_tokenized_string(),
                                                              add_special_tokens=True,
                                                              max_length=512,
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

                index_of_CLS_token = 0 if self.initial_cls_token else len(subtokens) -1

                cls_embeddings_all_layers: List[torch.FloatTensor] = \
                    [hidden_states[layer][sentence_idx][index_of_CLS_token] for layer in self.layer_indexes]

                # use scalar mix of embeddings if so selected
                if self.use_scalar_mix:
                    sm = ScalarMix(mixture_size=len(cls_embeddings_all_layers))
                    sm_embeddings = sm(cls_embeddings_all_layers)

                    cls_embeddings_all_layers = [sm_embeddings]

                # set the extracted embedding for the token
                sentence.set_embedding(self.name, torch.cat(cls_embeddings_all_layers))

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.use_scalar_mix
            else self.model.config.hidden_size
        )

    def __setstate__(self, d):
        self.__dict__ = d

        # reload tokenizer to get around serialization issues
        model_name = self.name.split('transformer-document-')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


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