import inspect
import logging
import os
import random
import re
import tempfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from typing import Dict, Generic, List, Optional, Sequence, Union

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
from flair.data import DT, Sentence, Token

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

    def _get_processed_token_text(self, token: Token) -> str:
        pieces = self.tokenizer.tokenize(token.text)
        token_text = ""
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

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

    def _reconstruct_tokens_from_subtokens(self, sentence, subtokens):
        word_iterator = iter(sentence)
        token = next(word_iterator)
        token_text = self._get_processed_token_text(token)
        token_subtoken_lengths = []
        reconstructed_token = ""
        subtoken_count = 0
        # iterate over subtokens and reconstruct tokens
        for subtoken_id, subtoken in enumerate(subtokens):

            # remove special markup
            subtoken = self._remove_special_markup(subtoken)

            # TODO check if this is necessary is this method is called before prepare_for_model
            # check if reconstructed token is special begin token ([CLS] or similar)
            if subtoken in self.special_tokens and subtoken_id == 0:
                continue

            # some BERT tokenizers somehow omit words - in such cases skip to next token
            if subtoken_count == 0 and not token_text.startswith(subtoken.lower()):

                while True:
                    token_subtoken_lengths.append(0)
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                    if token_text.startswith(subtoken.lower()):
                        break

            subtoken_count += 1

            # append subtoken to reconstruct token
            reconstructed_token = reconstructed_token + subtoken

            # check if reconstructed token is the same as current token
            if reconstructed_token.lower() == token_text:

                # if so, add subtoken count
                token_subtoken_lengths.append(subtoken_count)

                # reset subtoken count and reconstructed token
                reconstructed_token = ""
                subtoken_count = 0

                # break from loop if all tokens are accounted for
                if len(token_subtoken_lengths) < len(sentence):
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                else:
                    break

        # if tokens are unaccounted for
        while len(token_subtoken_lengths) < len(sentence) and len(token.text) == 1:
            token_subtoken_lengths.append(0)
            if len(token_subtoken_lengths) == len(sentence):
                break
            token = next(word_iterator)

        # check if all tokens were matched to subtokens
        if token != sentence[-1]:
            log.error(f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'")
            log.error(f"Last matched: '{token}'")
            log.error(f"Last sentence: '{sentence[-1]}'")
            log.error(f"subtokenized: '{subtokens}'")
        return token_subtoken_lengths

    def _gather_flair_tokens(self, sentences: List[Sentence]):
        sentence_tokens = []
        for sentence in sentences:
            # flair specific pre-tokenization
            tokens = [token.text for token in sentence]
            sentence_tokens.append(tokens)
        return sentence_tokens

    def _build_transformer_model_inputs(self, batch_encoding, sentences):
        input_ids = batch_encoding["input_ids"].to(flair.device)
        model_kwargs = {"input_ids": input_ids}

        # Models such as FNet do not have an attention_mask
        if "attention_mask" in batch_encoding:
            model_kwargs["attention_mask"] = batch_encoding["attention_mask"].to(flair.device)

        # set language IDs for XLM-style transformers
        if self.use_lang_emb and self.tokenizer.lang2id is not None:
            model_kwargs["langs"] = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            if not self.allow_long_sentences:
                for s_id, sentence in enumerate(sentences):
                    lang_id = self.tokenizer.lang2id.get(sentence.get_language_code(), 0)
                    model_kwargs["langs"][s_id] = lang_id
            else:
                sentence_part_lengths = torch.unique(
                    batch_encoding["overflow_to_sample_mapping"],
                    return_counts=True,
                    sorted=True,
                )[1].tolist()
                sentence_idx = 0
                for sentence, part_length in zip(sentences, sentence_part_lengths):
                    lang_id = self.tokenizer.lang2id.get(sentence.get_language_code(), 0)
                    model_kwargs["langs"][sentence_idx : sentence_idx + part_length] = lang_id
                    sentence_idx += part_length
        word_ids = [
            torch.tensor([-100 if val is None else val for val in batch_encoding.word_ids(i)], device=flair.device)
            for i, _ in enumerate(sentences)
        ]
        model_kwargs["word_ids"] = word_ids
        return model_kwargs

    def _combine_strided_sentences(
        self, hidden_states: torch.Tensor, overflow_to_sample_mapping: torch.Tensor
    ) -> List[torch.Tensor]:
        sentence_hidden_states = []
        for sentence_id in torch.arange(0, overflow_to_sample_mapping.max() + 1):
            selected_sentences = hidden_states[overflow_to_sample_mapping == sentence_id]
            if selected_sentences.size()[0] == 1:
                sentence_hidden_states.append(selected_sentences[0, :])
            else:
                sentence_hidden_states.append(
                    torch.cat(
                        [selected_sentences[0, : self.stride // 2]]
                        + list(selected_sentences[:, 1 + self.stride // 2 : hidden_states.size()[1] - 1 - self.stride // 2])
                        + [
                            selected_sentences[
                                selected_sentences.size()[0] - 1, hidden_states.size()[1] - 1 - self.stride // 2 :
                            ]
                        ]
                    )
                )
        return sentence_hidden_states

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
        self, word_ids: List[torch.Tensor], overflow_to_sample_mapping: Optional[torch.Tensor] = None, **kwargs
    ):
        hidden_states = self.model(**kwargs)[-1]

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
            sentence_hidden_states = self._combine_strided_sentences(
                hidden_states,
                overflow_to_sample_mapping,
            )
        else:
            sentence_hidden_states = list(hidden_states)

        result = dict()

        if self.document_embedding:
            document_embeddings = []
            for sentence_hidden_state in sentence_hidden_states:
                if self.cls_pooling == "cls":
                    index_of_cls_token = 0 if self.initial_cls_token else sentence_hidden_state.size()[0] - 1
                    document_emb = sentence_hidden_state[index_of_cls_token, :]
                elif self.cls_pooling == "mean":
                    document_emb = sentence_hidden_state.mean(dim=0)
                elif self.cls_pooling == "max":
                    document_emb, _ = sentence_hidden_state.max(dim=0)
                else:
                    raise ValueError(f"cls pooling method: `{self.cls_pooling}` is not implemented")
                document_embeddings.append(document_emb)
            result["document_embeddings"] = document_embeddings

        if self.token_embedding:
            all_token_embeddings = []
            for sentence_hidden_state, word_id in zip(sentence_hidden_states, word_ids):
                if self.subtoken_pooling == "first":
                    first_mask = torch.cat(
                        [torch.tensor([True], device=flair.device), word_id[1:] != word_id[: word_id.size()[0] - 1]]
                    )
                    token_embeddings = sentence_hidden_state[(word_id >= 0) & first_mask, :]
                elif self.subtoken_pooling == "last":
                    last_mask = torch.cat(
                        [word_id[1:] != word_id[: word_id.size()[0] - 1], torch.tensor([True], device=flair.device)]
                    )
                    token_embeddings = sentence_hidden_state[(word_id >= 0) & last_mask]
                elif self.subtoken_pooling == "first_last":
                    first_mask = torch.cat([[True], word_id[1:] != word_id[: word_id.size()[0] - 1]])
                    last_mask = torch.cat(
                        [word_id[1:] != word_id[: word_id.size()[0] - 1], torch.tensor([True], device=flair.device)]
                    )
                    first_token_embeddings = sentence_hidden_state[(word_id >= 0) & first_mask]
                    last_token_embeddings = sentence_hidden_state[(word_id >= 0) & last_mask]
                    token_embeddings = torch.cat([first_token_embeddings, last_token_embeddings], dim=1)
                elif self.subtoken_pooling == "mean":
                    tokens = []
                    for _id in torch.arange(word_id.max() + 1):
                        tokens.append(sentence_hidden_state[word_id == _id].mean(dim=1))
                    token_embeddings = torch.cat(tokens, dim=0)
                else:
                    raise ValueError(f"subtoken pooling method: `{self.subtoken_pooling}` is not implemented")
                all_token_embeddings.append(token_embeddings)
            result["token_embeddings"] = all_token_embeddings
        return result

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        flair_tokens = self._gather_flair_tokens(sentences)

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

        model_kwargs = self._build_transformer_model_inputs(batch_encoding, sentences)

        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:
            embeddings = self.forward(
                overflow_to_sample_mapping=batch_encoding.get("overflow_to_sample_mapping"), **model_kwargs
            )

            if self.document_embedding:
                self._extract_document_embeddings(embeddings["document_embeddings"], sentences)

            if self.token_embedding:
                self._extract_token_embeddings(embeddings["token_embeddings"], sentences)

    def _expand_sentence_with_context(self, sentence):
        expand_context = not self.training or random.randint(1, 100) > (self.context_dropout * 100)

        left_context = []
        right_context = []

        if expand_context:
            left_context = sentence.left_context(self.context_length, self.respect_document_boundaries)
            right_context = sentence.right_context(self.context_length, self.respect_document_boundaries)

        expanded_sentence = Sentence(left_context + [t.text for t in sentence.tokens] + right_context)

        context_length = len(left_context)
        return expanded_sentence, context_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        expanded_sentences = []
        context_offsets = []

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

            for sentence in sentences:
                # create expanded sentence and remember context offsets
                expanded_sentence, context_offset = self._expand_sentence_with_context(sentence)
                expanded_sentences.append(expanded_sentence)
                context_offsets.append(context_offset)
        else:
            expanded_sentences.extend(sentences)

        self._add_embeddings_to_sentences(expanded_sentences)

        # move embeddings from context back to original sentence (if using context)
        if self.context_length > 0:
            for original_sentence, expanded_sentence, context_offset in zip(
                sentences, expanded_sentences, context_offsets
            ):
                if self.token_embedding:
                    for token_idx, token in enumerate(original_sentence):
                        token.set_embedding(
                            self.name,
                            expanded_sentence[token_idx + context_offset].get_embedding(self.name),
                        )
                if self.document_embedding:
                    original_sentence.set_embedding(self.name, expanded_sentence.get_embedding(self.name))
