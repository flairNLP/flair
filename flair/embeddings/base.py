import inspect
import logging
import os
import random
import re
import tempfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import Dict, Generic, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch.nn import Parameter, ParameterList
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import LARGE_INTEGER

import flair
from flair.data import DT, Sentence

log = logging.getLogger("flair")


class Embeddings(torch.nn.Module, Generic[DT]):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    def __init__(self):
        """Set some attributes that would otherwise result in errors. Overwrite these in your embedding class."""
        if not hasattr(self, "name"):
            self.name: str = "unnamed_embedding"
        if not hasattr(self, "static_embeddings"):
            # if the embeddings for a sentence are the same in each epoch, set this to True for improved efficiency
            self.static_embeddings = False
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        raise NotImplementedError

    def embed(self, data_points: Union[DT, List[DT]]) -> List[DT]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if not isinstance(data_points, list):
            data_points = [data_points]

        if not self._everything_embedded(data_points) or not self.static_embeddings:
            self._add_embeddings_internal(data_points)

        return data_points

    def _everything_embedded(self, data_points: Sequence[DT]) -> bool:
        for data_point in data_points:
            if self.name not in data_point._embeddings.keys():
                return False
        return True

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[DT]):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        return [self.name]

    def get_named_embeddings_dict(self) -> Dict:
        return {self.name: self}

    @staticmethod
    def get_instance_parameters(locals: dict) -> dict:
        class_definition = locals.get("__class__")
        instance_parameter_names = set(inspect.signature(class_definition.__init__).parameters)  # type: ignore
        instance_parameter_names.remove("self")
        instance_parameter_names.add("__class__")
        instance_parameters = {
            class_attribute: attribute_value
            for class_attribute, attribute_value in locals.items()
            if class_attribute in instance_parameter_names
        }
        return instance_parameters


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors.
    This method was proposed by Liu et al. (2019) in the paper:
    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)

    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.
    It can be found under:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.
    """

    def __init__(self, mixture_size: int, trainable: bool = False) -> None:
        """
        Inits scalar mix implementation.
        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
        :param mixture_size: size of mixtures (usually the number of layers)
        """
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size

        initial_scalar_parameters = [0.0] * mixture_size

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.tensor(
                        [initial_scalar_parameters[i]],
                        dtype=torch.float,
                        device=flair.device,
                    ),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(
            torch.tensor(
                [1.0],
                dtype=torch.float,
                device=flair.device,
            ),
            requires_grad=trainable,
        )

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error(
                "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
                    len(tensors), self.mixture_size
                )
            )

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights_split = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights_split, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


@torch.jit.script_if_tracing
def pad_sequence_embeddings(all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
    embedding_length = all_hidden_states[0].shape[1]
    longest_token_sequence_in_batch = 0
    for hidden_states in all_hidden_states:
        if hidden_states.shape[0] > longest_token_sequence_in_batch:
            longest_token_sequence_in_batch = hidden_states.shape[0]
    pre_allocated_zero_tensor = torch.zeros(
        embedding_length * longest_token_sequence_in_batch,
        dtype=torch.float,
        device=flair.device,
    )
    all_embs = []
    for hidden_states in all_hidden_states:
        all_embs.append(hidden_states.view(-1))
        nb_padding_tokens = longest_token_sequence_in_batch - hidden_states.shape[0]
        if nb_padding_tokens > 0:
            all_embs.append(pre_allocated_zero_tensor[: embedding_length * nb_padding_tokens])
    return torch.cat(all_embs).view(len(all_hidden_states), longest_token_sequence_in_batch, embedding_length)


@torch.jit.script_if_tracing
def combine_strided_tensors(
    hidden_states: torch.Tensor,
    overflow_to_sample_mapping: torch.Tensor,
    half_stride: int,
    max_length: int,
    default_value: int,
) -> torch.Tensor:
    _, counts = torch.unique(overflow_to_sample_mapping, sorted=True, return_counts=True)
    sentence_count = int(overflow_to_sample_mapping.max().item() + 1)
    token_count = max_length + (max_length - 2) * int(counts.max().item() - 1)
    if hidden_states.dim() == 2:
        sentence_hidden_states = torch.zeros(
            (sentence_count, token_count), device=flair.device, dtype=hidden_states.dtype
        )
    else:
        sentence_hidden_states = torch.zeros(
            (sentence_count, token_count, hidden_states.shape[2]), device=flair.device, dtype=hidden_states.dtype
        )

    sentence_hidden_states += default_value

    for sentence_id in torch.arange(0, sentence_hidden_states.shape[0]):
        selected_sentences = hidden_states[overflow_to_sample_mapping == sentence_id]
        start_part = selected_sentences[0, : half_stride + 1]
        mid_part = selected_sentences[:, half_stride + 1 : max_length - 1 - half_stride]
        mid_part = torch.reshape(mid_part, (mid_part.shape[0] * mid_part.shape[1],) + mid_part.shape[2:])
        end_part = selected_sentences[selected_sentences.shape[0] - 1, max_length - half_stride - 1 :]
        sentence_hidden_state = torch.cat((start_part, mid_part, end_part), dim=0)
        sentence_hidden_states[sentence_id, : sentence_hidden_state.shape[0]] = torch.cat(
            (start_part, mid_part, end_part), dim=0
        )

    return sentence_hidden_states


@torch.jit.script_if_tracing
def fill_masked_elements(
    all_token_embeddings: torch.Tensor, sentence_hidden_states: torch.Tensor, mask: torch.Tensor, word_ids: torch.Tensor
):
    for i in torch.arange(int(all_token_embeddings.shape[0])):
        all_token_embeddings[i, : int(word_ids[i].max()) + 1, :] = insert_missing_embeddings(
            sentence_hidden_states[i][mask[i] & (word_ids[i] >= 0)], word_ids[i]
        )
    return all_token_embeddings


@torch.jit.script_if_tracing
def insert_missing_embeddings(token_embeddings: torch.Tensor, word_id: torch.Tensor) -> torch.Tensor:
    # in some cases we need to insert zero vectors for tokens without embedding.
    if token_embeddings.shape[0] < word_id.max() + 1:
        for _id in torch.arange(int(word_id.max()) + 1):
            if not (word_id == _id).any():
                token_embeddings = torch.cat(
                    (
                        token_embeddings[:_id],
                        torch.zeros_like(token_embeddings[:1]),
                        token_embeddings[_id:],
                    ),
                    dim=0,
                )
    return token_embeddings


@torch.jit.script_if_tracing
def fill_mean_token_embeddings(
    all_token_embeddings: torch.Tensor, sentence_hidden_states: torch.Tensor, word_ids: torch.Tensor
):
    for i in torch.arange(all_token_embeddings.shape[0]):
        for _id in torch.arange(int(word_ids[i].max()) + 1):
            all_token_embeddings[i, _id, :] = torch.nan_to_num(
                sentence_hidden_states[i][word_ids[i] == _id].mean(dim=0)
            )
    return all_token_embeddings


@torch.jit.script_if_tracing
def document_mean_pooling(sentence_hidden_states: torch.Tensor, sentence_lengths: torch.Tensor):
    result = torch.zeros(sentence_hidden_states.shape[0], sentence_hidden_states.shape[2])

    for i in torch.arange(sentence_hidden_states.shape[0]):
        result[i] = sentence_hidden_states[i, : sentence_lengths[i]].mean(dim=0)  # type: ignore


@torch.jit.script_if_tracing
def document_max_pooling(sentence_hidden_states: torch.Tensor, sentence_lengths: torch.Tensor):
    result = torch.zeros(sentence_hidden_states.shape[0], sentence_hidden_states.shape[2])

    for i in torch.arange(sentence_hidden_states.shape[0]):
        result[i], _ = sentence_hidden_states[i, : sentence_lengths[i]].max(dim=0)  # type: ignore


class TransformerEmbedding(Embeddings[Sentence]):
    def __init__(
        self,
        model: str = "bert-base-uncased",
        fine_tune: bool = True,
        layers: str = "-1",
        layer_mean: bool = True,
        subtoken_pooling: str = "first",
        cls_pooling: str = "cls",
        is_token_embedding: bool = True,
        is_document_embedding: bool = True,
        allow_long_sentences: bool = False,
        use_context: Union[bool, int] = False,
        respect_document_boundaries: bool = True,
        context_dropout: float = 0.5,
        saved_config: Optional[PretrainedConfig] = None,
        tokenizer_data: Optional[BytesIO] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        del self.instance_parameters["saved_config"]
        del self.instance_parameters["tokenizer_data"]
        super().__init__()
        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # do not print transformer warnings as these are confusing in this case
        from transformers import logging

        logging.set_verbosity_error()

        if tokenizer_data is None:
            # load tokenizer and transformer model
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True, **kwargs)
        else:
            # load tokenizer from inmemory zip-file
            self.tokenizer = self._tokenizer_from_bytes(tokenizer_data)

        if saved_config is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
            self.model = AutoModel.from_pretrained(model, config=config)
        else:
            self.model = AutoModel.from_config(saved_config, **kwargs)

        self.truncate = True

        if self.tokenizer.model_max_length > LARGE_INTEGER:
            allow_long_sentences = False
            self.truncate = False

        self.stride = self.tokenizer.model_max_length // 2 if allow_long_sentences else 0
        self.allow_long_sentences = allow_long_sentences
        self.use_lang_emb = hasattr(self.model, "use_lang_emb") and self.model.use_lang_emb

        # model name
        if name is None:
            self.name = "transformer-" + str(model)
        else:
            self.name = name
        self.base_model_name = str(model)

        self.token_embedding = is_token_embedding
        self.document_embedding = is_document_embedding

        if not self.token_embedding and not self.document_embedding:
            raise ValueError("either 'is_token_embedding' or 'is_document_embedding' needs to be set.")

        if self.document_embedding and cls_pooling not in ["cls", "max", "mean"]:
            raise ValueError(f"Document Pooling operation `{cls_pooling}` is not defined for TransformerEmbedding")

        if self.token_embedding and subtoken_pooling not in ["first", "last", "first_last", "mean"]:
            raise ValueError(f"Subtoken Pooling operation `{subtoken_pooling}` is not defined for TransformerEmbedding")

        if self.document_embedding and cls_pooling == "cls" and allow_long_sentences:
            log.warning(
                "Using long sentences for Document embeddings is only beneficial for cls_pooling types 'mean' and 'max "
            )

        if isinstance(use_context, bool):
            self.context_length: int = 64 if use_context else 0
        else:
            self.context_length = use_context

        self.context_dropout = context_dropout
        self.respect_document_boundaries = respect_document_boundaries

        self.to(flair.device)

        # embedding parameters
        if layers == "all":
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = list(range(len(hidden_states)))
        else:
            self.layer_indexes = list(map(int, layers.split(",")))

        self.cls_pooling = cls_pooling
        self.subtoken_pooling = subtoken_pooling
        self.layer_mean = layer_mean
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune

        # return length
        self.embedding_length_internal = self._calculate_embedding_length()

        self.special_tokens = []
        # check if special tokens exist to circumvent error message
        self.has_unk = self.tokenizer._unk_token is not None
        if self.tokenizer._bos_token:
            self.special_tokens.append(self.tokenizer.bos_token)
        if self.tokenizer._cls_token:
            self.special_tokens.append(self.tokenizer.cls_token)

        # most models have an initial BOS token, except for XLNet, T5 and GPT2
        self.begin_offset = self._get_begin_offset_of_tokenizer()
        self.initial_cls_token: bool = self._has_initial_cls_token()

        # when initializing, embeddings are in eval mode by default
        self.eval()

    @property
    def embedding_length(self) -> int:
        if not hasattr(self, "embedding_length_internal"):
            self.embedding_length_internal = self._calculate_embedding_length()

        return self.embedding_length_internal

    def _has_initial_cls_token(self) -> bool:
        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        tokens = self.tokenizer.encode("a")
        return tokens[0] == self.tokenizer.cls_token_id

    def _get_begin_offset_of_tokenizer(self) -> int:
        test_string = "a"
        tokens_with_special = self.tokenizer.encode(test_string)
        tokens_without_special = self.tokenizer.encode(test_string, add_special_tokens=False)
        normal_count = len(tokens_without_special)

        for begin_offset in range(len(tokens_with_special) - normal_count):
            if tokens_with_special[begin_offset : begin_offset + normal_count] == tokens_without_special:
                return begin_offset
        log.warning(
            f"Could not determine the begin offset of the tokenizer for transformer model {self.name}, assuming 0"
        )
        return 0

    def _calculate_embedding_length(self) -> int:
        if not self.layer_mean:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size

        # in case of doubt: token embedding has higher priority than document embedding
        if self.token_embedding and self.subtoken_pooling == "first_last":
            length *= 2
            if self.document_embedding:
                log.warning(
                    "Token embedding length and Document embedding length vary, due to `first_last` subtoken pooling, this might not be supported"
                )
        return length

    @property
    def embedding_type(self) -> str:
        # in case of doubt: token embedding has higher priority than document embedding
        return "word-level" if self.token_embedding else "sentence-level"

    def _tokenizer_from_bytes(self, zip_data: BytesIO) -> PreTrainedTokenizer:
        zip_obj = zipfile.ZipFile(zip_data)
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_obj.extractall(temp_dir)
            return AutoTokenizer.from_pretrained(temp_dir, add_prefix_space=True)

    def _tokenizer_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files = list(self.tokenizer.save_pretrained(temp_dir))
            if self.tokenizer.is_fast:
                vocab_files = self.tokenizer.slow_tokenizer_class.vocab_files_names.values()
                files = [f for f in files if all(v not in f for v in vocab_files)]
            zip_data = BytesIO()
            zip_obj = zipfile.ZipFile(zip_data, "w")
            for f in files:
                # transformers returns the "added_tokens.json" even if it doesn't create it
                if os.path.exists(f):
                    zip_obj.write(f, os.path.relpath(f, temp_dir))

        zip_data.seek(0)
        return zip_data

    @staticmethod
    def _remove_special_markup(text: str):
        # remove special markup
        text = re.sub("^Ġ", "", text)  # RoBERTa models
        text = re.sub("^##", "", text)  # BERT models
        text = re.sub("^▁", "", text)  # XLNet models
        text = re.sub("</w>$", "", text)  # XLM models
        return text

    def _get_processed_token_text(self, token: str) -> str:
        pieces = self.tokenizer.tokenize(token)
        token_text = ""
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text.strip()

    def __getstate__(self):
        config_dict = self.model.config.to_dict()

        tokenizer_data = self._tokenizer_bytes()

        model_state = {
            "model": self.base_model_name,
            "fine_tune": self.fine_tune,
            "layers": ",".join(map(str, self.layer_indexes)),
            "layer_mean": self.layer_mean,
            "subtoken_pooling": self.subtoken_pooling,
            "cls_pooling": self.cls_pooling,
            "is_token_embedding": self.token_embedding,
            "is_document_embedding": self.document_embedding,
            "allow_long_sentences": self.allow_long_sentences,
            "config_state_dict": config_dict,
            "tokenizer_data": tokenizer_data,
            "name": self.name,
            "context_length": self.context_length,
            "respect_document_boundaries": self.respect_document_boundaries,
            "context_dropout": self.context_dropout,
        }

        return model_state

    def __setstate__(self, state):
        config_state_dict = state.pop("config_state_dict", None)
        model_state_dict = state.pop("model_state_dict", None)

        # legacy TransformerDocumentEmbedding
        state.pop("batch_size", None)
        state.pop("embedding_length_internal", None)
        # legacy TransformerTokenEmbedding
        state.pop("memory_effective_training", None)

        if "base_model_name" in state:
            state["model"] = state.pop("base_model_name")

        state["use_context"] = state.pop("context_length", False)

        if "layer_indexes" in state:
            layer_indexes = state.pop("layer_indexes")
            state["layers"] = ",".join(map(str, layer_indexes))

        if "use_scalar_mix" in state:
            # legacy Flair <= 0.7
            state["layer_mean"] = state.pop("use_scalar_mix")

        if "is_token_embedding" not in state:
            # legacy TransformerTokenEmbedding
            state["is_token_embedding"] = "pooling_operation" in state

        if "is_document_embedding" not in state:
            # Legacy TransformerDocumentEmbedding
            state["is_document_embedding"] = "pooling" in state

        if "pooling_operation" in state:
            # legacy TransformerTokenEmbedding
            state["subtoken_pooling"] = state.pop("pooling_operation")

        if "cls_operation" in state:
            # legacy TransformerDocumentEmbedding
            state["cls_pooling"] = state.pop("pooling")

        config = None

        if config_state_dict:
            model_type = config_state_dict.get("model_type", "bert")
            config_class = CONFIG_MAPPING[model_type]
            config = config_class.from_dict(config_state_dict)

        embedding = self.create_from_state(saved_config=config, **state)

        # copy values from new embedding
        for key in embedding.__dict__.keys():
            self.__dict__[key] = embedding.__dict__[key]

        if model_state_dict:
            self.model.load_state_dict(model_state_dict)

    @classmethod
    def create_from_state(cls, **state):
        return cls(**state)

    def _reconstruct_word_ids_from_subtokens(self, tokens: List[str], subtokens: List[str]):
        word_iterator = iter(enumerate(map(self._get_processed_token_text, tokens)))
        token_id, token_text = next(word_iterator)
        word_ids: List[Optional[int]] = []
        reconstructed_token = ""
        subtoken_count = 0
        processed_first_token = False
        # iterate over subtokens and reconstruct tokens
        for subtoken_id, subtoken in enumerate(subtokens):

            # remove special markup
            subtoken = self._remove_special_markup(subtoken)

            # check if reconstructed token is special begin token ([CLS] or similar)
            if subtoken in self.special_tokens:
                word_ids.append(None)
                continue

            if subtoken_count == 0 and processed_first_token:
                token_id, token_text = next(word_iterator)
            processed_first_token = True
            # some BERT tokenizers somehow omit words - in such cases skip to next token
            while subtoken_count == 0 and not token_text.startswith(subtoken.lower()):
                token_id, token_text = next(word_iterator)
            word_ids.append(token_id)
            subtoken_count += 1

            reconstructed_token = reconstructed_token + subtoken
            print(reconstructed_token)

            if reconstructed_token.lower() == token_text:
                # we cannot handle unk_tokens perfectly, so let's assume that one unk_token corresponds to one token.
                reconstructed_token = ""
                subtoken_count = 0

        # if tokens are unaccounted for
        while len(word_ids) < len(subtokens):
            word_ids.append(None)

        # check if all tokens were matched to subtokens
        if token_id + 1 != len(tokens) and not self.truncate:
            log.error(f"Reconstructed token: '{reconstructed_token}'")
            log.error(f"Tokenization MISMATCH in sentence '{' '.join(tokens)}'")
            log.error(f"Last matched: '{tokens[token_id]}'")
            log.error(f"Last sentence: '{tokens[-1]}'")
            log.error(f"subtokenized: '{subtokens}'")
        return word_ids

    def _gather_flair_tokens(self, sentences: List[Sentence]) -> Tuple[List[List[str]], List[int], List[int]]:
        offsets = []
        lengths = []
        if self.context_length > 0:
            # set context if not set already
            previous_sentence = None
            for sentence in sentences:
                if sentence.is_context_set():
                    continue
                sentence._previous_sentence = previous_sentence
                sentence._next_sentence = None
                if previous_sentence:
                    previous_sentence._next_sentence = sentence
                previous_sentence = sentence

        sentence_tokens = []
        for sentence in sentences:
            # flair specific pre-tokenization
            tokens, offset = self._expand_sentence_with_context(sentence)
            sentence_tokens.append(tokens)
            offsets.append(offset)
            lengths.append(len(sentence))
        return sentence_tokens, offsets, lengths

    def _build_transformer_model_inputs(
        self,
        batch_encoding,
        sentences: List[Sentence],
        offsets: List[int],
        lengths: List[int],
        flair_tokens: List[List[str]],
    ):
        input_ids = batch_encoding["input_ids"].to(flair.device, non_blocking=True)
        model_kwargs = {"input_ids": input_ids}

        # Models such as FNet do not have an attention_mask
        if "attention_mask" in batch_encoding:
            model_kwargs["attention_mask"] = batch_encoding["attention_mask"].to(flair.device, non_blocking=True)

        needs_length = self.document_embedding and not (self.cls_pooling == "cls" and self.initial_cls_token)

        if "overflow_to_sample_mapping" in batch_encoding:
            model_kwargs["overflow_to_sample_mapping"] = batch_encoding["overflow_to_sample_mapping"].to(flair.device, non_blocking=True)
            if needs_length:
                unpacked_ids = combine_strided_tensors(
                    input_ids,
                    model_kwargs["overflow_to_sample_mapping"],
                    self.stride // 2,
                    self.tokenizer.model_max_length,
                    self.tokenizer.pad_token_id,
                )
                lengths = (unpacked_ids != self.tokenizer.pad_token_id).sum(dim=1)
                model_kwargs["lengths"] = lengths
        else:
            if needs_length:
                lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
                model_kwargs["lengths"] = lengths

        # set language IDs for XLM-style transformers
        if self.use_lang_emb and getattr(self.tokenizer, "lang2id") is not None:
            model_kwargs["langs"] = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            lang2id = getattr(self.tokenizer, "lang2id")
            if not self.allow_long_sentences:
                for s_id, sentence in enumerate(sentences):
                    lang_id = lang2id.get(sentence.get_language_code(), 0)
                    model_kwargs["langs"][s_id] = lang_id
            else:
                sentence_part_lengths = torch.unique(
                    batch_encoding["overflow_to_sample_mapping"],
                    return_counts=True,
                    sorted=True,
                )[1].tolist()
                sentence_idx = 0
                for sentence, part_length in zip(sentences, sentence_part_lengths):
                    lang_id = lang2id.get(sentence.get_language_code(), 0)
                    model_kwargs["langs"][sentence_idx : sentence_idx + part_length] = lang_id
                    sentence_idx += part_length
        if self.token_embedding:
            if self.tokenizer.is_fast:
                word_ids_list = [batch_encoding.word_ids(i) for i in range(input_ids.size()[0])]
            else:
                # word_ids is only supported for the fast rust tokenizers. Some models like "xlm-mlm-ende-1024" do not have
                # a fast tokenizer implementation, hence we need to fall back to our own reconstruction of word_ids.
                word_ids_list = []
                max_len = 0
                for tokens in flair_tokens:
                    token_texts = self.tokenizer.tokenize(" ".join(tokens), is_split_into_words=True)
                    token_ids = cast(List[int], self.tokenizer.convert_tokens_to_ids(token_texts))
                    expanded_token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
                    j = 0
                    for i, token_id in enumerate(token_ids):
                        while expanded_token_ids[j] != token_id:
                            token_texts.insert(j, self.tokenizer.convert_ids_to_tokens(expanded_token_ids[j]))
                            j += 1
                        j += 1
                    while j < len(expanded_token_ids):
                        token_texts.insert(j, self.tokenizer.convert_ids_to_tokens(expanded_token_ids[j]))
                        j += 1
                    if not self.allow_long_sentences and self.truncate:
                        token_texts = token_texts[: self.tokenizer.model_max_length]
                    reconstruct = self._reconstruct_word_ids_from_subtokens(tokens, token_texts)
                    word_ids_list.append(reconstruct)
                    reconstruct_len = len(reconstruct)
                    if reconstruct_len > max_len:
                        max_len = reconstruct_len
                for _word_ids in word_ids_list:
                    # padding
                    _word_ids.extend([None] * (max_len - len(_word_ids)))

            if self.allow_long_sentences:
                new_offsets = []
                new_lengths = []
                for sent_id in batch_encoding["overflow_to_sample_mapping"]:
                    new_offsets.append(offsets[sent_id])
                    new_lengths.append(lengths[sent_id])
                offsets = new_offsets
                lengths = new_lengths

            word_ids = torch.tensor(
                [
                    [
                        -100 if (val is None or val < offset or val >= offset + length) else val - offset
                        for val in _word_ids
                    ]
                    for _word_ids, offset, length in zip(word_ids_list, offsets, lengths)
                ],
                device=flair.device,
            )
            model_kwargs["word_ids"] = word_ids
        return model_kwargs

    def _can_document_embedding_shortcut(self):
        # cls first pooling can be done without recreating sentence hidden states
        return (
            self.document_embedding
            and not self.token_embedding
            and self.cls_pooling == "cls"
            and self.initial_cls_token
        )

    def _extract_document_embeddings(self, sentence_hidden_states, sentences):
        for document_emb, sentence in zip(sentence_hidden_states, sentences):
            sentence.set_embedding(self.name, document_emb)

    def _extract_token_embeddings(self, sentence_embeddings, sentences):
        for token_embeddings, sentence in zip(sentence_embeddings, sentences):
            for token_embedding, token in zip(token_embeddings, sentence):
                token.set_embedding(self.name, token_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        overflow_to_sample_mapping: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
    ):
        model_kwargs = {}
        if langs is not None:
            model_kwargs["langs"] = langs
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        hidden_states = self.model(input_ids, **model_kwargs)[-1]

        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)

        # only use layers that will be outputted
        hidden_states = hidden_states[self.layer_indexes, :, :]
        if self.layer_mean:
            hidden_states = hidden_states.mean(dim=0)
        else:
            hidden_states = torch.flatten(hidden_states.permute((0, 3, 1, 2)), 0, 1).permute((1, 2, 0))

        if self._can_document_embedding_shortcut():
            return {"document_embeddings": hidden_states[:, 0]}

        if self.allow_long_sentences:
            assert overflow_to_sample_mapping is not None
            sentence_hidden_states = combine_strided_tensors(
                hidden_states, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, 0
            )
            if self.tokenizer.is_fast and self.token_embedding:
                word_ids = combine_strided_tensors(
                    word_ids, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, -100
                )
        else:
            sentence_hidden_states = hidden_states

        result = dict()

        if self.document_embedding:
            if self.cls_pooling == "cls" and self.initial_cls_token:
                document_embeddings = sentence_hidden_states[:, 0]
            else:
                assert lengths is not None
                if self.cls_pooling == "cls":
                    document_embeddings = sentence_hidden_states[
                        torch.arange(sentence_hidden_states.shape[0]), lengths - 1
                    ]
                elif self.cls_pooling == "mean":
                    document_embeddings = document_mean_pooling(sentence_hidden_states, lengths)
                elif self.cls_pooling == "max":
                    document_embeddings = document_max_pooling(sentence_hidden_states, lengths)
                else:
                    raise ValueError(f"cls pooling method: `{self.cls_pooling}` is not implemented")
            result["document_embeddings"] = document_embeddings

        if self.token_embedding:
            assert word_ids is not None
            all_token_embeddings = torch.zeros(
                word_ids.shape[0], word_ids.max() + 1, self.embedding_length_internal, device=flair.device
            )  # type: ignore
            true_tensor = torch.ones_like(word_ids[:, :1], dtype=torch.bool)
            if self.subtoken_pooling == "first":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                first_mask = torch.cat([true_tensor, gain_mask], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, first_mask, word_ids
                )
            elif self.subtoken_pooling == "last":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                last_mask = torch.cat([gain_mask, true_tensor], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, last_mask, word_ids
                )
            elif self.subtoken_pooling == "first_last":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                first_mask = torch.cat([true_tensor, gain_mask], dim=1)
                last_mask = torch.cat([gain_mask, true_tensor], dim=1)
                all_token_embeddings[:, :, : sentence_hidden_states.shape[2]] = fill_masked_elements(
                    all_token_embeddings[:, :, : sentence_hidden_states.shape[2]],
                    sentence_hidden_states,
                    first_mask,
                    word_ids,
                )
                all_token_embeddings[:, :, sentence_hidden_states.shape[2] :] = fill_masked_elements(
                    all_token_embeddings[:, :, sentence_hidden_states.shape[2] :],
                    sentence_hidden_states,
                    last_mask,
                    word_ids,
                )
            elif self.subtoken_pooling == "mean":
                all_token_embeddings = fill_mean_token_embeddings(
                    all_token_embeddings, sentence_hidden_states, word_ids
                )
            else:
                raise ValueError(f"subtoken pooling method: `{self.subtoken_pooling}` is not implemented")

            result["token_embeddings"] = all_token_embeddings
        return result

    def _prepare_tensors(self, sentences: List[Sentence]):
        flair_tokens, offsets, lengths = self._gather_flair_tokens(sentences)

        # encode inputs
        batch_encoding = self.tokenizer(
            flair_tokens,
            stride=self.stride,
            return_overflowing_tokens=self.allow_long_sentences,
            truncation=self.truncate,
            padding=True,
            return_tensors="pt",
            is_split_into_words=True,
        )

        forward_kwargs = self._build_transformer_model_inputs(batch_encoding, sentences, offsets, lengths, flair_tokens)

        return forward_kwargs

    def _expand_sentence_with_context(self, sentence) -> Tuple[List[str], int]:
        expand_context = self.context_length > 0 and (
            not self.training or random.randint(1, 100) > (self.context_dropout * 100)
        )

        left_context = []
        right_context = []

        if expand_context:
            left_context = sentence.left_context(self.context_length, self.respect_document_boundaries)
            right_context = sentence.right_context(self.context_length, self.respect_document_boundaries)

        expanded_sentence = left_context + [t.text for t in sentence.tokens] + right_context

        context_length = len(left_context)
        return expanded_sentence, context_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        tensors = self._prepare_tensors(sentences)
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:
            embeddings = self.forward(**tensors)

            if self.document_embedding:
                self._extract_document_embeddings(embeddings["document_embeddings"], sentences)

            if self.token_embedding:
                self._extract_token_embeddings(embeddings["token_embeddings"], sentences)
