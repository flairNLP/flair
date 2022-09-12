import inspect
import os
import random
import re
import tempfile
import zipfile
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch.jit import ScriptModule
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import LARGE_INTEGER
from transformers.utils import PaddingStrategy

import flair
from flair.data import Sentence, log
from flair.embeddings.base import DocumentEmbeddings, Embeddings, TokenEmbeddings


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
        if selected_sentences.shape[0] > 1:
            start_part = selected_sentences[0, : half_stride + 1]
            mid_part = selected_sentences[:, half_stride + 1 : max_length - 1 - half_stride]
            mid_part = torch.reshape(mid_part, (mid_part.shape[0] * mid_part.shape[1],) + mid_part.shape[2:])
            end_part = selected_sentences[selected_sentences.shape[0] - 1, max_length - half_stride - 1 :]
            sentence_hidden_state = torch.cat((start_part, mid_part, end_part), dim=0)
            sentence_hidden_states[sentence_id, : sentence_hidden_state.shape[0]] = torch.cat(
                (start_part, mid_part, end_part), dim=0
            )
        else:
            sentence_hidden_states[sentence_id, : selected_sentences.shape[1]] = selected_sentences[0, :]

    return sentence_hidden_states


@torch.jit.script_if_tracing
def fill_masked_elements(
    all_token_embeddings: torch.Tensor,
    sentence_hidden_states: torch.Tensor,
    mask: torch.Tensor,
    word_ids: torch.Tensor,
    lengths: torch.LongTensor,
):
    for i in torch.arange(int(all_token_embeddings.shape[0])):
        all_token_embeddings[i, : lengths[i], :] = insert_missing_embeddings(  # type: ignore
            sentence_hidden_states[i][mask[i] & (word_ids[i] >= 0)], word_ids[i], lengths[i]
        )
    return all_token_embeddings


@torch.jit.script_if_tracing
def insert_missing_embeddings(
    token_embeddings: torch.Tensor, word_id: torch.Tensor, length: torch.LongTensor
) -> torch.Tensor:
    # in some cases we need to insert zero vectors for tokens without embedding.
    if token_embeddings.shape[0] < length:
        for _id in torch.arange(int(length)):
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


def _legacy_reconstruct_word_ids(
    embedding: "TransformerBaseEmbeddings", flair_tokens: List[List[str]]
) -> List[List[Optional[int]]]:
    word_ids_list = []
    max_len = 0
    for tokens in flair_tokens:
        token_texts = embedding.tokenizer.tokenize(" ".join(tokens), is_split_into_words=True)
        token_ids = cast(List[int], embedding.tokenizer.convert_tokens_to_ids(token_texts))
        expanded_token_ids = embedding.tokenizer.build_inputs_with_special_tokens(token_ids)
        j = 0
        for i, token_id in enumerate(token_ids):
            while expanded_token_ids[j] != token_id:
                token_texts.insert(j, embedding.tokenizer.convert_ids_to_tokens(expanded_token_ids[j]))
                j += 1
            j += 1
        while j < len(expanded_token_ids):
            token_texts.insert(j, embedding.tokenizer.convert_ids_to_tokens(expanded_token_ids[j]))
            j += 1
        if not embedding.allow_long_sentences and embedding.truncate:
            token_texts = token_texts[: embedding.tokenizer.model_max_length]
        reconstruct = _reconstruct_word_ids_from_subtokens(embedding, tokens, token_texts)
        word_ids_list.append(reconstruct)
        reconstruct_len = len(reconstruct)
        if reconstruct_len > max_len:
            max_len = reconstruct_len
    for _word_ids in word_ids_list:
        # padding
        _word_ids.extend([None] * (max_len - len(_word_ids)))
    return word_ids_list


def remove_special_markup(text: str):
    # remove special markup
    text = re.sub("^Ġ", "", text)  # RoBERTa models
    text = re.sub("^##", "", text)  # BERT models
    text = re.sub("^▁", "", text)  # XLNet models
    text = re.sub("</w>$", "", text)  # XLM models
    return text


def _get_processed_token_text(tokenizer, token: str) -> str:
    pieces = tokenizer.tokenize(token)
    token_text = "".join(map(remove_special_markup, pieces))
    token_text = token_text.lower()
    return token_text.strip()


def _reconstruct_word_ids_from_subtokens(embedding, tokens: List[str], subtokens: List[str]):
    word_iterator = iter(enumerate((_get_processed_token_text(embedding.tokenizer, token) for token in tokens)))
    token_id, token_text = next(word_iterator)
    word_ids: List[Optional[int]] = []
    reconstructed_token = ""
    subtoken_count = 0
    processed_first_token = False

    special_tokens = []
    # check if special tokens exist to circumvent error message
    if embedding.tokenizer._bos_token:
        special_tokens.append(embedding.tokenizer.bos_token)
    if embedding.tokenizer._cls_token:
        special_tokens.append(embedding.tokenizer.cls_token)

    # iterate over subtokens and reconstruct tokens
    for subtoken_id, subtoken in enumerate(subtokens):

        # remove special markup
        subtoken = remove_special_markup(subtoken)

        # check if reconstructed token is special begin token ([CLS] or similar)
        if subtoken in special_tokens:
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
    if token_id + 1 != len(tokens) and not embedding.truncate:
        log.error(f"Reconstructed token: '{reconstructed_token}'")
        log.error(f"Tokenization MISMATCH in sentence '{' '.join(tokens)}'")
        log.error(f"Last matched: '{tokens[token_id]}'")
        log.error(f"Last sentence: '{tokens[-1]}'")
        log.error(f"subtokenized: '{subtokens}'")
    return word_ids


class TransformerBaseEmbeddings(Embeddings[Sentence]):
    """Base class for all TransformerEmbeddings
    This base class handles the tokenizer and the input preparation, however it won't implement the actual model.
    This can be further extended to implement the model in either a pytorch, jit or onnx way of working.
    """

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        embedding_length: int,
        context_length: int,
        context_dropout: float,
        respect_document_boundaries: bool,
        stride: int,
        allow_long_sentences: bool,
        fine_tune: bool,
        truncate: bool,
        use_lang_emb: bool,
        document_embedding: bool = False,
        token_embedding: bool = False,
        force_device: Optional[torch.device] = None,
        force_max_length: bool = False,
    ):
        self.name = name
        super().__init__()
        self.document_embedding = document_embedding
        self.token_embedding = token_embedding
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.embedding_length_internal = embedding_length
        self.context_length = context_length
        self.context_dropout = context_dropout
        self.respect_document_boundaries = respect_document_boundaries
        self.stride = stride
        self.allow_long_sentences = allow_long_sentences
        self.truncate = truncate
        self.use_lang_emb = use_lang_emb
        self.force_device = force_device
        self.fine_tune = fine_tune
        self.force_max_length = force_max_length

        if not self.token_embedding and not self.document_embedding:
            raise ValueError("either 'is_token_embedding' or 'is_document_embedding' needs to be set.")

    def to_args(self):
        return {
            "token_embedding": self.token_embedding,
            "document_embedding": self.document_embedding,
            "allow_long_sentences": self.allow_long_sentences,
            "tokenizer": self.tokenizer,
            "context_length": self.context_length,
            "context_dropout": self.context_dropout,
            "respect_document_boundaries": self.respect_document_boundaries,
            "truncate": self.truncate,
            "stride": self.stride,
            "embedding_length": self.embedding_length_internal,
            "name": self.name,
            "fine_tune": self.fine_tune,
            "use_lang_emb": self.use_lang_emb,
            "force_max_length": self.force_max_length,
        }

    def __getstate__(self):
        model_state = self.to_args()
        del model_state["tokenizer"]
        model_state["tokenizer_data"] = self.__tokenizer_bytes()

        return model_state

    def __setstate__(self, state):
        tokenizer = self._tokenizer_from_bytes(state.pop("tokenizer_data"))
        embedding = self.create_from_state(tokenizer=tokenizer, **state)
        for key in embedding.__dict__.keys():
            self.__dict__[key] = embedding.__dict__[key]

    def _tokenizer_from_bytes(self, zip_data: BytesIO) -> PreTrainedTokenizer:
        zip_obj = zipfile.ZipFile(zip_data)
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_obj.extractall(temp_dir)
            return AutoTokenizer.from_pretrained(temp_dir, add_prefix_space=True)

    def __tokenizer_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files = list(self.tokenizer.save_pretrained(temp_dir))
            if self.tokenizer.is_fast and self.tokenizer.slow_tokenizer_class:
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

    @classmethod
    def create_from_state(cls, **state):
        return cls(**state)

    @property
    def embedding_length(self) -> int:
        return self.embedding_length_internal

    @property
    def embedding_type(self) -> str:
        # in case of doubt: token embedding has higher priority than document embedding
        return "word-level" if self.token_embedding else "sentence-level"

    @abstractmethod
    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        return self(**tensors)

    def prepare_tensors(self, sentences: List[Sentence], device: torch.device = None):
        if device is None:
            device = flair.device
        flair_tokens, offsets, lengths = self.__gather_flair_tokens(sentences)

        # encode inputs
        batch_encoding = self.tokenizer(
            flair_tokens,
            stride=self.stride,
            return_overflowing_tokens=self.allow_long_sentences,
            truncation=self.truncate,
            padding=PaddingStrategy.MAX_LENGTH if self.force_max_length else PaddingStrategy.LONGEST,
            return_tensors="pt",
            is_split_into_words=True,
        )

        tensor_args = self.__build_transformer_model_inputs(
            batch_encoding, sentences, offsets, lengths, flair_tokens, device
        )

        return tensor_args

    def __build_transformer_model_inputs(
        self,
        batch_encoding,
        sentences: List[Sentence],
        offsets: List[int],
        sentence_lengths: List[int],
        flair_tokens: List[List[str]],
        device: torch.device,
    ):
        input_ids = batch_encoding["input_ids"].to(device, non_blocking=True)
        model_kwargs = {"input_ids": input_ids}

        # Models such as FNet do not have an attention_mask
        if "attention_mask" in batch_encoding:
            model_kwargs["attention_mask"] = batch_encoding["attention_mask"].to(device, non_blocking=True)

        if "overflow_to_sample_mapping" in batch_encoding:
            cpu_overflow_to_sample_mapping = batch_encoding["overflow_to_sample_mapping"]
            model_kwargs["overflow_to_sample_mapping"] = cpu_overflow_to_sample_mapping.to(device, non_blocking=True)
            unpacked_ids = combine_strided_tensors(
                input_ids,
                model_kwargs["overflow_to_sample_mapping"],
                self.stride // 2,
                self.tokenizer.model_max_length,
                self.tokenizer.pad_token_id,
            )
            lengths = (unpacked_ids != self.tokenizer.pad_token_id).sum(dim=1)
        else:
            cpu_overflow_to_sample_mapping = None
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
                word_ids_list = _legacy_reconstruct_word_ids(
                    self,
                    flair_tokens,
                )
                # word_ids is only supported for the fast rust tokenizers. Some models like "xlm-mlm-ende-1024" do not have
                # a fast tokenizer implementation, hence we need to fall back to our own reconstruction of word_ids.

            if self.allow_long_sentences:
                new_offsets = []
                new_lengths = []
                assert cpu_overflow_to_sample_mapping is not None
                for sent_id in cpu_overflow_to_sample_mapping:
                    new_offsets.append(offsets[sent_id])
                    new_lengths.append(sentence_lengths[sent_id])
                offsets = new_offsets
                sentence_lengths = new_lengths

            word_ids = torch.tensor(
                [
                    [
                        -100 if (val is None or val < offset or val >= offset + length) else val - offset
                        for val in _word_ids
                    ]
                    for _word_ids, offset, length in zip(word_ids_list, offsets, sentence_lengths)
                ],
                device=device,
            )
            model_kwargs["word_ids"] = word_ids

        # Ensure that the keys in the dict are in the proper order expected from the model forward function
        desired_keys_order = [
            "input_ids",
            "lengths",
            "attention_mask",
            "overflow_to_sample_mapping",
            "word_ids",
            "langs",
        ]
        model_kwargs = {k: model_kwargs[k] for k in desired_keys_order if k in model_kwargs}

        return model_kwargs

    def __gather_flair_tokens(self, sentences: List[Sentence]) -> Tuple[List[List[str]], List[int], List[int]]:
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
            tokens, offset = self.__expand_sentence_with_context(sentence)
            sentence_tokens.append(tokens)
            offsets.append(offset)
            lengths.append(len(sentence))
        return sentence_tokens, offsets, lengths

    def __expand_sentence_with_context(self, sentence) -> Tuple[List[str], int]:
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

    def __extract_document_embeddings(self, sentence_hidden_states, sentences):
        for document_emb, sentence in zip(sentence_hidden_states, sentences):
            sentence.set_embedding(self.name, document_emb)

    def __extract_token_embeddings(self, sentence_embeddings, sentences):
        for token_embeddings, sentence in zip(sentence_embeddings, sentences):
            for token_embedding, token in zip(token_embeddings, sentence):
                token.set_embedding(self.name, token_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        tensors = self.prepare_tensors(sentences, device=self.force_device)
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()
        with gradient_context:
            embeddings = self._forward_tensors(tensors)

        if self.document_embedding:
            document_embedding = embeddings["document_embeddings"]
            self.__extract_document_embeddings(document_embedding, sentences)

        if self.token_embedding:
            token_embedding = embeddings["token_embeddings"]
            self.__extract_token_embeddings(token_embedding, sentences)


class TransformerOnnxEmbeddings(TransformerBaseEmbeddings):
    def __init__(self, onnx_model: str, providers: List = [], **kwargs):
        # onnx prepares numpy arrays, no mather if it runs on gpu or cpu, the input is on cpu first.
        super().__init__(**kwargs, force_device=torch.device("cpu"))
        self.onnx_model = onnx_model
        self.providers = providers
        self.create_session()
        self.eval()

    def __getstate__(self):
        state = super().__getstate__()
        state["providers"] = self.providers
        state["onnx_model"] = self.onnx_model
        return state

    def create_session(self):
        try:
            import onnxruntime
        except ImportError:
            log.error(
                "You cannot use OnnxEmbeddings without ONNXruntime being installed,"
                "please run `pip install onnxruntime`"
            )
            raise
        if os.path.isfile(self.onnx_model):
            self.session = onnxruntime.InferenceSession(self.onnx_model, providers=self.providers)
        else:
            log.warning(
                f"Could not find file '{self.onnx_model}' used in {self.__class__.name}({self.name})."
                "The embedding won't work unless a valid path is set."
            )
            self.session = None

    def remove_session(self):
        if self.session is not None:
            self.session._sess = None
            del self.session
        self.session = None

    def optimize_model(self, optimize_model_path, use_external_data_format: bool = False, **kwargs):
        """Wrapper for onnxruntime.transformers.optimizer.optimize_model"""
        from onnxruntime.transformers.optimizer import optimize_model

        self.remove_session()
        model = optimize_model(self.onnx_model, **kwargs)
        model.save_model_to_file(optimize_model_path, use_external_data_format=use_external_data_format)
        self.onnx_model = optimize_model_path
        self.create_session()

    def quantize_model(self, quantize_model_path, use_external_data_format: bool = False, **kwargs):
        from onnxruntime.quantization import quantize_dynamic

        self.remove_session()
        quantize_dynamic(
            self.onnx_model, quantize_model_path, use_external_data_format=use_external_data_format, **kwargs
        )
        self.onnx_model = quantize_model_path
        self.create_session()

    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        input_array = {k: v.numpy() for k, v in tensors.items()}
        embeddings = self.session.run([], input_array)

        result = dict()
        if self.document_embedding:
            result["document_embeddings"] = torch.tensor(embeddings[0], device=flair.device)
        if self.token_embedding:
            result["token_embeddings"] = torch.tensor(embeddings[-1], device=flair.device)

        return result

    @classmethod
    def collect_dynamic_axes(cls, embedding: "TransformerEmbeddings", tensors):
        dynamic_axes = dict()
        for k, v in tensors.items():
            if k == "lengths":
                dynamic_axes[k] = {0: "sent-count"}
                continue
            if k == "word_ids":
                if embedding.tokenizer.is_fast:
                    dynamic_axes[k] = {0: "batch", 1: "sequ_length"}
                else:
                    dynamic_axes[k] = {0: "sent-count", 1: "max_token_count"}
                continue
            if k == "overflow_to_sample_mapping":
                dynamic_axes[k] = {0: "batch"}
            if v.dim() == 1:
                dynamic_axes[k] = {0: "batch"}
            else:
                dynamic_axes[k] = {0: "batch", 1: "sequ_length"}
        if embedding.token_embedding:
            dynamic_axes["token_embeddings"] = {0: "sent-count", 1: "max_token_count", 2: "token_embedding_size"}
        if embedding.document_embedding:
            dynamic_axes["document_embeddings"] = {0: "sent-count", 1: "document_embedding_size"}
        return dynamic_axes

    @classmethod
    def export_from_embedding(
        cls,
        path: Union[str, Path],
        embedding: "TransformerEmbeddings",
        example_sentences: List[Sentence],
        opset_version: int = 13,
        providers: List = None,
    ):
        path = str(path)
        example_tensors = embedding.prepare_tensors(example_sentences)
        dynamic_axes = cls.collect_dynamic_axes(embedding, example_tensors)
        output_names = []
        if embedding.document_embedding:
            output_names.append("document_embeddings")
        if embedding.token_embedding:
            output_names.append("token_embeddings")

        if providers is None:
            if flair.device.type == "cuda":
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]

        torch.onnx.export(
            embedding,
            [example_tensors],
            path,
            input_names=list(example_tensors.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
        return cls(onnx_model=path, providers=providers, **embedding.to_args())


class TransformerJitEmbeddings(TransformerBaseEmbeddings):
    def __init__(self, jit_model: Union[bytes, ScriptModule], param_names: List[str], **kwargs):
        super().__init__(**kwargs)
        if isinstance(jit_model, bytes):
            buffer = BytesIO(jit_model)
            buffer.seek(0)
            self.jit_model: ScriptModule = torch.jit.load(buffer, map_location=flair.device)
        else:
            self.jit_model = jit_model
        self.param_names = param_names

        self.to(flair.device)
        self.eval()

    def __getstate__(self):
        state = super().__getstate__()
        buffer = BytesIO()
        torch.jit.save(self.jit_model, buffer)
        state["jit_model"] = buffer.getvalue()
        state["param_names"] = self.param_names
        return state

    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        parameters = []
        for param in self.param_names:
            parameters.append(tensors[param])
        embeddings = self.jit_model(*parameters)
        if isinstance(embeddings, tuple):
            return {"document_embeddings": embeddings[0], "token_embeddings": embeddings[1]}
        elif self.token_embedding:
            return {"token_embeddings": embeddings}
        elif self.document_embedding:
            return {"document_embeddings": embeddings}
        else:
            raise ValueError("either 'token_embedding' or 'document_embedding' needs to be set.")

    @classmethod
    def create_from_embedding(cls, module: ScriptModule, embedding: "TransformerEmbeddings", param_names: List[str]):
        return cls(jit_model=module, param_names=param_names, **embedding.to_args())

    @classmethod
    def parameter_to_list(
        cls, embedding: "TransformerEmbeddings", wrapper: torch.nn.Module, sentences: List[Sentence]
    ) -> Tuple[List[str], List[torch.Tensor]]:
        tensors = embedding.prepare_tensors(sentences)
        param_names = list(inspect.signature(wrapper.forward).parameters.keys())
        params = []
        for param in param_names:
            params.append(tensors[param])
        return param_names, params


class TorchWrapper(torch.nn.Module):
    def __init__(self, embedding: TransformerBaseEmbeddings):
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.LongTensor,
        attention_mask: torch.Tensor,
        overflow_to_sample_mapping: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding.forward(
            input_ids=input_ids,
            lengths=lengths,
            attention_mask=attention_mask,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
            word_ids=word_ids,
        )["token_embeddings"]


class TransformerNebullvmEmbeddings(TransformerBaseEmbeddings):
    def __init__(self, nebullvm_model, **kwargs):
        super().__init__(**kwargs)
        self.model = nebullvm_model

    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        return {"token_embeddings": self.model(*tensors.values())[0]}

    @classmethod
    def convert_dynamic_axes(cls, dynamic_info: Dict[str, Dict[int, str]]) -> Dict[str, List[Dict[int, str]]]:
        dynamic_axes: Dict[str, List[Dict[int, str]]] = dict()
        dynamic_axes["inputs"] = []
        dynamic_axes["outputs"] = []
        for k, v in dynamic_info.items():
            if k not in ["token_embeddings", "document_embeddings"]:
                dynamic_axes["inputs"].append(v)
            else:
                dynamic_axes["outputs"].append(v)

        return dynamic_axes

    @classmethod
    def optimize_model(
        cls,
        embedding: "TransformerEmbeddings",
        example_sentences: List[Sentence],
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
    ):
        example_tensors = embedding.prepare_tensors(example_sentences)

        if dynamic_info is None:
            dynamic_axes = TransformerOnnxEmbeddings.collect_dynamic_axes(embedding, example_tensors)
            dynamic_info = cls.convert_dynamic_axes(dynamic_axes)

        input_data = [(tuple(example_tensors.values()), 0)]
        torch_wrapper = TorchWrapper(embedding)
        torch_wrapper.eval()

        # Import nebullvm only if it is used
        from nebullvm.api.functions import optimize_model

        optimized_model = optimize_model(
            torch_wrapper,
            input_data=input_data,
            metric_drop_ths=metric_drop_ths,
            metric=metric,
            optimization_time=optimization_time,
            dynamic_info=dynamic_info,
            config_file=config_file,
            ignore_compilers=ignore_compilers,
        )

        return cls(optimized_model, **embedding.to_args())


class TransformerJitWordEmbeddings(TokenEmbeddings, TransformerJitEmbeddings):
    def __init__(
        self,
        **kwargs,
    ):
        TransformerJitEmbeddings.__init__(self, **kwargs)


class TransformerJitDocumentEmbeddings(DocumentEmbeddings, TransformerJitEmbeddings):
    def __init__(
        self,
        **kwargs,
    ):
        TransformerJitEmbeddings.__init__(self, **kwargs)


class TransformerOnnxWordEmbeddings(TokenEmbeddings, TransformerOnnxEmbeddings):
    def __init__(
        self,
        **kwargs,
    ):
        TransformerOnnxEmbeddings.__init__(self, **kwargs)


class TransformerOnnxDocumentEmbeddings(DocumentEmbeddings, TransformerOnnxEmbeddings):
    def __init__(
        self,
        **kwargs,
    ):
        TransformerOnnxEmbeddings.__init__(self, **kwargs)


class TransformerEmbeddings(TransformerBaseEmbeddings):
    onnx_cls: Type[TransformerOnnxEmbeddings] = TransformerOnnxEmbeddings
    nebullvm_cls: Type[TransformerNebullvmEmbeddings] = TransformerNebullvmEmbeddings

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
        force_max_length: bool = False,
        **kwargs,
    ):
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        del self.instance_parameters["saved_config"]
        del self.instance_parameters["tokenizer_data"]
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

        def is_supported_t5_model(config: PretrainedConfig) -> bool:
            t5_supported_model_types = ["t5", "mt5", "longt5"]
            return getattr(config, "model_type", "") in t5_supported_model_types

        if saved_config is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)

            if is_supported_t5_model(config):
                from transformers import T5EncoderModel

                transformer_model = T5EncoderModel.from_pretrained(model, config=config)
            else:
                transformer_model = AutoModel.from_pretrained(model, config=config)
        else:
            if is_supported_t5_model(saved_config):
                from transformers import T5EncoderModel

                transformer_model = T5EncoderModel(saved_config, **kwargs)
            else:
                transformer_model = AutoModel.from_config(saved_config, **kwargs)
        transformer_model = transformer_model.to(flair.device)

        self.truncate = True
        self.force_max_length = force_max_length

        if self.tokenizer.model_max_length > LARGE_INTEGER:
            allow_long_sentences = False
            self.truncate = False

        self.stride = self.tokenizer.model_max_length // 2 if allow_long_sentences else 0
        self.allow_long_sentences = allow_long_sentences
        self.use_lang_emb = hasattr(transformer_model, "use_lang_emb") and transformer_model.use_lang_emb

        # model name
        if name is None:
            self.name = "transformer-" + str(transformer_model)
        else:
            self.name = name
        self.base_model_name = str(transformer_model)

        self.token_embedding = is_token_embedding
        self.document_embedding = is_document_embedding

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

        # embedding parameters
        if layers == "all":
            # send mini-token through to check how many layers the model has
            hidden_states = transformer_model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = list(range(len(hidden_states)))
        else:
            self.layer_indexes = list(map(int, layers.split(",")))

        self.cls_pooling = cls_pooling
        self.subtoken_pooling = subtoken_pooling
        self.layer_mean = layer_mean
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune

        # return length
        self.embedding_length_internal = self._calculate_embedding_length(transformer_model)

        # most models have an initial BOS token, except for XLNet, T5 and GPT2
        self.initial_cls_token: bool = self._has_initial_cls_token()

        super().__init__(**self.to_args())
        self.model = transformer_model
        self.to(flair.device)
        # when initializing, embeddings are in eval mode by default
        self.eval()

    @property
    def embedding_length(self) -> int:
        if not hasattr(self, "embedding_length_internal"):
            self.embedding_length_internal = self._calculate_embedding_length(self.model)

        return self.embedding_length_internal

    def _has_initial_cls_token(self) -> bool:
        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        tokens = self.tokenizer.encode("a")
        return tokens[0] == self.tokenizer.cls_token_id

    def _calculate_embedding_length(self, model) -> int:
        if not self.layer_mean:
            length = len(self.layer_indexes) * model.config.hidden_size
        else:
            length = model.config.hidden_size

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

    def __getstate__(self):
        config_dict = self.model.config.to_dict()
        super_state = super().__getstate__()
        tokenizer_data = super_state["tokenizer_data"]

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
            "force_max_length": self.force_max_length,
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

        if "pooling" in state:
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

    def _can_document_embedding_shortcut(self):
        # cls first pooling can be done without recreating sentence hidden states
        return (
            self.document_embedding
            and not self.token_embedding
            and self.cls_pooling == "cls"
            and self.initial_cls_token
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        overflow_to_sample_mapping: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
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
                word_ids.shape[0], int(lengths.max()), self.embedding_length_internal, device=flair.device
            )
            true_tensor = torch.ones_like(word_ids[:, :1], dtype=torch.bool)
            if self.subtoken_pooling == "first":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                first_mask = torch.cat([true_tensor, gain_mask], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, first_mask, word_ids, lengths
                )
            elif self.subtoken_pooling == "last":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                last_mask = torch.cat([gain_mask, true_tensor], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, last_mask, word_ids, lengths
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
                    lengths,
                )
                all_token_embeddings[:, :, sentence_hidden_states.shape[2] :] = fill_masked_elements(
                    all_token_embeddings[:, :, sentence_hidden_states.shape[2] :],
                    sentence_hidden_states,
                    last_mask,
                    word_ids,
                    lengths,
                )
            elif self.subtoken_pooling == "mean":
                all_token_embeddings = fill_mean_token_embeddings(
                    all_token_embeddings, sentence_hidden_states, word_ids, lengths
                )
            else:
                raise ValueError(f"subtoken pooling method: `{self.subtoken_pooling}` is not implemented")

            result["token_embeddings"] = all_token_embeddings
        return result

    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        return self.forward(**tensors)

    def export_onnx(self, path: str, example_sentences: List[Sentence], **kwargs) -> TransformerOnnxEmbeddings:
        """
        Export TransformerEmbeddings to OnnxFormat.
        :param example_sentences: a list of sentences that will be used for tracing. It is recommended to take 2-4
        sentences with some variation.
        """
        return self.onnx_cls.export_from_embedding(path, self, example_sentences, **kwargs)

    def optimize_nebullvm(
        self,
        example_sentences: List[Sentence],
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
    ):
        """Optimize the input model using the nebullvm api.
        :param example_sentences: a list of sentences that will be used for tracing. It is recommended to take 2-4
        sentences with some variation.
        Info about other optional params can be found in the nebullvm documentation:
        https://nebuly.gitbook.io/nebuly/nebulgym/get-started
        """
        return self.nebullvm_cls.optimize_model(
            self,
            example_sentences,
            metric_drop_ths,
            metric,
            optimization_time,
            dynamic_info,
            config_file,
            ignore_compilers,
        )
