import hashlib
import logging
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from deprecated.sphinx import deprecated
from sentencepiece import SentencePieceProcessor
from torch import nn

import flair
from flair.class_utils import lazy_import
from flair.data import Corpus, Dictionary, Sentence, _iter_dataset
from flair.embeddings.base import TokenEmbeddings, load_embeddings, register_embeddings
from flair.embeddings.transformer import (
    TransformerEmbeddings,
    TransformerOnnxWordEmbeddings,
)
from flair.file_utils import cached_path, extract_single_zip_file, instance_lru_cache

log = logging.getLogger("flair")


@register_embeddings
class TransformerWordEmbeddings(TokenEmbeddings, TransformerEmbeddings):
    onnx_cls = TransformerOnnxWordEmbeddings

    def __init__(
        self,
        model: str = "bert-base-uncased",
        is_document_embedding: bool = False,
        allow_long_sentences: bool = True,
        **kwargs,
    ) -> None:
        """Bidirectional transformer embeddings of words from various transformer architectures.

        Args:
            model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for options)
            is_document_embedding: If True, the embedding can be used as DocumentEmbedding too.
            allow_long_sentences: If True, too long sentences will be patched and strided and afterwards combined.
            **kwargs: Arguments propagated to :meth:`flair.embeddings.transformer.TransformerEmbeddings.__init__`
        """
        TransformerEmbeddings.__init__(
            self,
            model=model,
            is_token_embedding=True,
            is_document_embedding=is_document_embedding,
            allow_long_sentences=allow_long_sentences,
            **kwargs,
        )

    @classmethod
    def create_from_state(cls, **state):
        # this parameter is fixed
        del state["is_token_embedding"]
        return cls(**state)


@register_embeddings
class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: list[TokenEmbeddings], overwrite_names: bool = True) -> None:
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            if overwrite_names:
                embedding.name = f"{i!s}-{embedding.name}"
            self.add_module(f"list_embedding_{i!s}", embedding)

        self.name: str = "Stack"
        self.__names = [name for embedding in self.embeddings for name in embedding.get_names()]

        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length
        self.eval()

    def embed(self, sentences: Union[Sentence, list[Sentence]], static_embeddings: bool = True):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self) -> str:
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> list[str]:
        """Returns a list of embedding names.

        In most cases, it is just a list with one item, namely the name of this embedding. But in some cases, the
        embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack.
        """
        # make compatible with serialized models
        if "__names" not in self.__dict__:
            self.__names = [name for embedding in self.embeddings for name in embedding.get_names()]

        return self.__names

    @classmethod
    def from_params(cls, params):
        embeddings = [load_embeddings(p) for p in params["embeddings"]]
        return cls(embeddings=embeddings, overwrite_names=False)

    def to_params(self):
        return {"embeddings": [emb.save_embeddings(use_state_dict=False) for emb in self.embeddings]}


@register_embeddings
class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(
        self,
        embeddings: Optional[str],
        field: Optional[str] = None,
        fine_tune: bool = False,
        force_cpu: bool = True,
        stable: bool = False,
        no_header: bool = False,
        vocab: Optional[dict[str, int]] = None,
        embedding_length: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes classic word embeddings.

        Constructor downloads required files if not there.

        Note:
            When loading a new embedding, you need to have `flair[gensim]` installed.

        Args:
            embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or a path to a custom embedding
            field: if given, the word-embeddings embed the data for the specific label-type instead of the plain text.
            fine_tune: If True, allows word-embeddings to be fine-tuned during training
            force_cpu: If True, stores the large embedding matrix not on the gpu to save memory. `force_cpu` can only be used if `fine_tune` is False
            stable: if True, use the stable embeddings as described in https://arxiv.org/abs/2110.02861
            no_header: only for reading plain word2vec text files. If true, the reader assumes the first line to not contain embedding length and vocab size.
            vocab: If the embeddings are already loaded in memory, provide the vocab here.
            embedding_length: If the embeddings are already loaded in memory, provide the embedding_length here.
            name: The name of the embeddings.
        """
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        if fine_tune and force_cpu and flair.device.type != "cpu":
            raise ValueError("Cannot train WordEmbeddings on cpu if the model is trained on gpu, set force_cpu=False")

        embeddings_path = self.resolve_precomputed_path(embeddings)
        if name is None:
            name = str(embeddings_path)

        self.name = name
        self.embeddings = embeddings if embeddings is not None else name
        self.static_embeddings = not fine_tune
        self.fine_tune = fine_tune
        self.force_cpu = force_cpu
        self.field = field
        self.stable = stable
        super().__init__()

        if embeddings_path is not None:
            (KeyedVectors,) = lazy_import("word-embeddings", "gensim.models", "KeyedVectors")
            if embeddings_path.suffix in [".bin", ".txt"]:
                precomputed_word_embeddings = KeyedVectors.load_word2vec_format(
                    str(embeddings_path), binary=embeddings_path.suffix == ".bin", no_header=no_header
                )
            else:
                precomputed_word_embeddings = KeyedVectors.load(str(embeddings_path))

            self.__embedding_length: int = precomputed_word_embeddings.vector_size

            vectors = np.vstack(
                (
                    precomputed_word_embeddings.vectors,
                    np.zeros(self.__embedding_length, dtype="float"),
                )
            )

            try:
                # gensim version 4
                self.vocab = precomputed_word_embeddings.key_to_index
            except AttributeError:
                # gensim version 3
                self.vocab = {k: v.index for k, v in precomputed_word_embeddings.vocab.items()}
        else:
            # if no embedding is set, the vocab and embedding length is required
            assert vocab is not None
            assert embedding_length is not None
            self.vocab = vocab
            self.__embedding_length = embedding_length
            vectors = np.zeros((len(self.vocab) + 1, self.__embedding_length), dtype="float")

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vectors), freeze=not fine_tune)

        if stable:
            self.layer_norm: Optional[nn.LayerNorm] = nn.LayerNorm(
                self.__embedding_length, elementwise_affine=fine_tune
            )
        else:
            self.layer_norm = None

        self.device = None
        self.to(flair.device)
        self.eval()

    def resolve_precomputed_path(self, embeddings: Optional[str]) -> Optional[Path]:
        if embeddings is None:
            return None

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/token"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{hu_path}/glove.gensim.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/glove.gensim", cache_dir=cache_dir)

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(f"{hu_path}/turian.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/turian", cache_dir=cache_dir)

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(f"{hu_path}/extvec.gensim.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/extvec.gensim", cache_dir=cache_dir)

        # pubmed embeddings
        elif embeddings.lower() == "pubmed" or embeddings.lower() == "en-pubmed":
            cached_path(
                f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim.vectors.npy",
                cache_dir=cache_dir,
            )
            return cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() in ["news", "en-news", "en"]:
            cached_path(f"{hu_path}/en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/en-fasttext-news-300d-1M", cache_dir=cache_dir)

        # twitter embeddings
        elif embeddings.lower() in ["twitter", "en-twitter"]:
            cached_path(f"{hu_path}/twitter.gensim.vectors.npy", cache_dir=cache_dir)
            return cached_path(f"{hu_path}/twitter.gensim", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(
                f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            return cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            return cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            return cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            return Path(embeddings)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @instance_lru_cache(maxsize=100000, typed=False)
    def get_cached_token_index(self, word: str) -> int:
        if word in self.vocab:
            return self.vocab[word]
        elif word.lower() in self.vocab:
            return self.vocab[word.lower()]
        elif re.sub(r"\d", "#", word.lower()) in self.vocab:
            return self.vocab[re.sub(r"\d", "#", word.lower())]
        elif re.sub(r"\d", "0", word.lower()) in self.vocab:
            return self.vocab[re.sub(r"\d", "0", word.lower())]
        else:
            return len(self.vocab)  # <unk> token

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        tokens = [token for sentence in sentences for token in sentence.tokens]

        word_indices: list[int] = []
        for token in tokens:
            word = token.text if self.field is None else token.get_label(self.field).value
            word_indices.append(self.get_cached_token_index(word))

        embeddings = self.embedding(torch.tensor(word_indices, dtype=torch.long, device=self.device))
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        if self.force_cpu:
            embeddings = embeddings.to(flair.device)

        for emb, token in zip(embeddings, tokens):
            token.set_embedding(self.name, emb)

        return sentences

    def __str__(self) -> str:
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"

    def train(self, mode=True):
        super().train(self.fine_tune and mode)

    def to(self, device):
        if self.force_cpu:
            device = torch.device("cpu")
        self.device = device
        super().to(device)

    def _apply(self, fn):
        if fn.__name__ == "convert" and self.force_cpu:
            # this is required to force the module on the cpu,
            # if a parent module is put to gpu, the _apply is called to each sub_module
            # self.to(..) actually sets the device properly
            if not hasattr(self, "device"):
                self.to(flair.device)
            return
        super()._apply(fn)

    def __getattribute__(self, item):
        # this ignores the get_cached_vec method when loading older versions
        # it is needed for compatibility reasons
        if item == "get_cached_vec":
            return None
        return super().__getattribute__(item)

    def __setstate__(self, state: dict[str, Any]):
        state.pop("get_cached_vec", None)
        state.setdefault("embeddings", state["name"])
        state.setdefault("force_cpu", True)
        state.setdefault("fine_tune", False)
        state.setdefault("field", None)
        if "precomputed_word_embeddings" in state:
            precomputed_word_embeddings = state.pop("precomputed_word_embeddings")
            vectors = np.vstack(
                (
                    precomputed_word_embeddings.vectors,
                    np.zeros(precomputed_word_embeddings.vector_size, dtype="float"),
                )
            )
            embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vectors), freeze=not state["fine_tune"])

            try:
                # gensim version 4
                vocab = precomputed_word_embeddings.key_to_index
            except AttributeError:
                # gensim version 3
                vocab = {k: v.index for k, v in precomputed_word_embeddings.__dict__["vocab"].items()}
            state["embedding"] = embedding
            state["vocab"] = vocab
        if "stable" not in state:
            state["stable"] = False
            state["layer_norm"] = None
        super().__setstate__(state)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "WordEmbeddings":
        return cls(embeddings=None, **params)

    def to_params(self) -> dict[str, Any]:
        return {
            "vocab": self.vocab,
            "stable": self.stable,
            "fine_tune": self.fine_tune,
            "force_cpu": self.force_cpu,
            "field": self.field,
            "name": self.name,
            "embedding_length": self.__embedding_length,
        }

    def state_dict(self, *args, **kwargs):
        # when loading the old versions from pickle, the embeddings might not be added as pytorch module.
        # we do this delayed, when the weights are collected (e.g. for saving), as doing this earlier might
        # lead to issues while loading (trying to load weights that weren't stored as python weights and therefore
        # not finding them)
        if list(self.modules()) == [self]:
            self.embedding = self.embedding
        return super().state_dict(*args, **kwargs)


@register_embeddings
class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        path_to_char_dict: Optional[Union[str, Dictionary]] = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
        name: str = "Char",
    ) -> None:
        """Instantiates a bidirectional lstm layer toi encode words by their character representation.

        Uses the default character dictionary if none provided.
        """
        super().__init__()
        self.name = name
        self.static_embeddings = False
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        elif isinstance(path_to_char_dict, Dictionary):
            self.char_dictionary = path_to_char_dict
        else:
            self.char_dictionary = Dictionary.load_from_file(path_to_char_dict)

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.hidden_size_char * 2

        self.to(flair.device)
        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        for sentence in sentences:
            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                char_indices = [self.char_dictionary.get_idx_for_item(char) for char in token.text]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(tokens_char_indices, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj:
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence),
                dtype=torch.long,
                device=flair.device,
            )

            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, : chars2_length[i]] = torch.tensor(c, dtype=torch.long, device=flair.device)

            # chars for rnn processing
            chars = tokens_mask

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(character_embeddings, chars2_length)

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)),
                dtype=outputs.dtype,
                device=flair.device,
            )
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "CharacterEmbeddings":
        return cls(**params)

    def to_params(self) -> dict[str, Any]:
        return {
            "path_to_char_dict": self.char_dictionary,
            "char_embedding_dim": self.char_embedding_dim,
            "hidden_size_char": self.hidden_size_char,
            "name": self.name,
        }


@register_embeddings
class FlairEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(
        self,
        model,
        fine_tune: bool = False,
        chars_per_chunk: int = 512,
        with_whitespace: bool = True,
        tokenized_lm: bool = True,
        is_lower: bool = False,
        name: Optional[str] = None,
        has_decoder: bool = False,
    ) -> None:
        """Initializes contextual string embeddings using a character-level language model.

        Args:
            model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
              'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward' depending on which character language model is desired.
            fine_tune: if set to True, the gradient will propagate into the language model.
              This dramatically slows down training and often leads to overfitting, so use with caution.
            chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff.
              Higher means faster but requires more memory. Lower means slower but less memory.
            with_whitespace: If True, use hidden state after whitespace after word.
              If False, use hidden state at last character of word.
            tokenized_lm: Whether this lm is tokenized. Default is True,
              but for LMs trained over unprocessed text False might be better.
            has_decoder: Weather to load the decoder-head of the LanguageModel. This should only be true, if you intend
              to generate text.
            is_lower: Whether this lm is trained on lower-cased data.
            name: The name of the embeddings
        """
        super().__init__()
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        cache_dir = Path("embeddings")

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/flair"
        clef_hipe_path: str = "https://files.ifi.uzh.ch/cl/siclemat/impresso/clef-hipe-2020/flair"
        am_path: str = "http://ltdata1.informatik.uni-hamburg.de/amharic/models/flair/"

        self.is_lower: bool = is_lower

        self.PRETRAINED_MODEL_ARCHIVE_MAP = {
            # multilingual models
            "multi-forward": f"{hu_path}/lm-jw300-forward-v0.1.pt",
            "multi-backward": f"{hu_path}/lm-jw300-backward-v0.1.pt",
            "multi-v0-forward": f"{hu_path}/lm-multi-forward-v0.1.pt",
            "multi-v0-backward": f"{hu_path}/lm-multi-backward-v0.1.pt",
            "multi-forward-fast": f"{hu_path}/lm-multi-forward-fast-v0.1.pt",
            "multi-backward-fast": f"{hu_path}/lm-multi-backward-fast-v0.1.pt",
            # English models
            "en-forward": f"{hu_path}/news-forward-0.4.1.pt",
            "en-backward": f"{hu_path}/news-backward-0.4.1.pt",
            "en-forward-fast": f"{hu_path}/lm-news-english-forward-1024-v0.2rc.pt",
            "en-backward-fast": f"{hu_path}/lm-news-english-backward-1024-v0.2rc.pt",
            "news-forward": f"{hu_path}/news-forward-0.4.1.pt",
            "news-backward": f"{hu_path}/news-backward-0.4.1.pt",
            "news-forward-fast": f"{hu_path}/lm-news-english-forward-1024-v0.2rc.pt",
            "news-backward-fast": f"{hu_path}/lm-news-english-backward-1024-v0.2rc.pt",
            "mix-forward": f"{hu_path}/lm-mix-english-forward-v0.2rc.pt",
            "mix-backward": f"{hu_path}/lm-mix-english-backward-v0.2rc.pt",
            # Arabic
            "ar-forward": f"{hu_path}/lm-ar-opus-large-forward-v0.1.pt",
            "ar-backward": f"{hu_path}/lm-ar-opus-large-backward-v0.1.pt",
            # Bulgarian
            "bg-forward-fast": f"{hu_path}/lm-bg-small-forward-v0.1.pt",
            "bg-backward-fast": f"{hu_path}/lm-bg-small-backward-v0.1.pt",
            "bg-forward": f"{hu_path}/lm-bg-opus-large-forward-v0.1.pt",
            "bg-backward": f"{hu_path}/lm-bg-opus-large-backward-v0.1.pt",
            # Czech
            "cs-forward": f"{hu_path}/lm-cs-opus-large-forward-v0.1.pt",
            "cs-backward": f"{hu_path}/lm-cs-opus-large-backward-v0.1.pt",
            "cs-v0-forward": f"{hu_path}/lm-cs-large-forward-v0.1.pt",
            "cs-v0-backward": f"{hu_path}/lm-cs-large-backward-v0.1.pt",
            # Danish
            "da-forward": f"{hu_path}/lm-da-opus-large-forward-v0.1.pt",
            "da-backward": f"{hu_path}/lm-da-opus-large-backward-v0.1.pt",
            # German
            "de-forward": f"{hu_path}/lm-mix-german-forward-v0.2rc.pt",
            "de-backward": f"{hu_path}/lm-mix-german-backward-v0.2rc.pt",
            "de-historic-ha-forward": f"{hu_path}/lm-historic-hamburger-anzeiger-forward-v0.1.pt",
            "de-historic-ha-backward": f"{hu_path}/lm-historic-hamburger-anzeiger-backward-v0.1.pt",
            "de-historic-wz-forward": f"{hu_path}/lm-historic-wiener-zeitung-forward-v0.1.pt",
            "de-historic-wz-backward": f"{hu_path}/lm-historic-wiener-zeitung-backward-v0.1.pt",
            "de-historic-rw-forward": f"{hu_path}/redewiedergabe_lm_forward.pt",
            "de-historic-rw-backward": f"{hu_path}/redewiedergabe_lm_backward.pt",
            # Spanish
            "es-forward": f"{hu_path}/lm-es-forward.pt",
            "es-backward": f"{hu_path}/lm-es-backward.pt",
            "es-forward-fast": f"{hu_path}/lm-es-forward-fast.pt",
            "es-backward-fast": f"{hu_path}/lm-es-backward-fast.pt",
            # Basque
            "eu-forward": f"{hu_path}/lm-eu-opus-large-forward-v0.2.pt",
            "eu-backward": f"{hu_path}/lm-eu-opus-large-backward-v0.2.pt",
            "eu-v1-forward": f"{hu_path}/lm-eu-opus-large-forward-v0.1.pt",
            "eu-v1-backward": f"{hu_path}/lm-eu-opus-large-backward-v0.1.pt",
            "eu-v0-forward": f"{hu_path}/lm-eu-large-forward-v0.1.pt",
            "eu-v0-backward": f"{hu_path}/lm-eu-large-backward-v0.1.pt",
            # Persian
            "fa-forward": f"{hu_path}/lm-fa-opus-large-forward-v0.1.pt",
            "fa-backward": f"{hu_path}/lm-fa-opus-large-backward-v0.1.pt",
            # Finnish
            "fi-forward": f"{hu_path}/lm-fi-opus-large-forward-v0.1.pt",
            "fi-backward": f"{hu_path}/lm-fi-opus-large-backward-v0.1.pt",
            # French
            "fr-forward": f"{hu_path}/lm-fr-charlm-forward.pt",
            "fr-backward": f"{hu_path}/lm-fr-charlm-backward.pt",
            # Hebrew
            "he-forward": f"{hu_path}/lm-he-opus-large-forward-v0.1.pt",
            "he-backward": f"{hu_path}/lm-he-opus-large-backward-v0.1.pt",
            # Hindi
            "hi-forward": f"{hu_path}/lm-hi-opus-large-forward-v0.1.pt",
            "hi-backward": f"{hu_path}/lm-hi-opus-large-backward-v0.1.pt",
            # Croatian
            "hr-forward": f"{hu_path}/lm-hr-opus-large-forward-v0.1.pt",
            "hr-backward": f"{hu_path}/lm-hr-opus-large-backward-v0.1.pt",
            # Indonesian
            "id-forward": f"{hu_path}/lm-id-opus-large-forward-v0.1.pt",
            "id-backward": f"{hu_path}/lm-id-opus-large-backward-v0.1.pt",
            # Italian
            "it-forward": f"{hu_path}/lm-it-opus-large-forward-v0.1.pt",
            "it-backward": f"{hu_path}/lm-it-opus-large-backward-v0.1.pt",
            # Japanese
            "ja-forward": f"{hu_path}/japanese-forward.pt",
            "ja-backward": f"{hu_path}/japanese-backward.pt",
            # Malayalam
            "ml-forward": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-forward.pt",
            "ml-backward": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-backward.pt",
            # Dutch
            "nl-forward": f"{hu_path}/lm-nl-opus-large-forward-v0.1.pt",
            "nl-backward": f"{hu_path}/lm-nl-opus-large-backward-v0.1.pt",
            "nl-v0-forward": f"{hu_path}/lm-nl-large-forward-v0.1.pt",
            "nl-v0-backward": f"{hu_path}/lm-nl-large-backward-v0.1.pt",
            # Norwegian
            "no-forward": f"{hu_path}/lm-no-opus-large-forward-v0.1.pt",
            "no-backward": f"{hu_path}/lm-no-opus-large-backward-v0.1.pt",
            # Polish
            "pl-forward": f"{hu_path}/lm-polish-forward-v0.2.pt",
            "pl-backward": f"{hu_path}/lm-polish-backward-v0.2.pt",
            "pl-opus-forward": f"{hu_path}/lm-pl-opus-large-forward-v0.1.pt",
            "pl-opus-backward": f"{hu_path}/lm-pl-opus-large-backward-v0.1.pt",
            # Portuguese
            "pt-forward": f"{hu_path}/lm-pt-forward.pt",
            "pt-backward": f"{hu_path}/lm-pt-backward.pt",
            # Pubmed
            "pubmed-forward": f"{hu_path}/pubmed-forward.pt",
            "pubmed-backward": f"{hu_path}/pubmed-backward.pt",
            "pubmed-2015-forward": f"{hu_path}/pubmed-2015-fw-lm.pt",
            "pubmed-2015-backward": f"{hu_path}/pubmed-2015-bw-lm.pt",
            # Slovenian
            "sl-forward": f"{hu_path}/lm-sl-opus-large-forward-v0.1.pt",
            "sl-backward": f"{hu_path}/lm-sl-opus-large-backward-v0.1.pt",
            "sl-v0-forward": f"{hu_path}/lm-sl-large-forward-v0.1.pt",
            "sl-v0-backward": f"{hu_path}/lm-sl-large-backward-v0.1.pt",
            # Swedish
            "sv-forward": f"{hu_path}/lm-sv-opus-large-forward-v0.1.pt",
            "sv-backward": f"{hu_path}/lm-sv-opus-large-backward-v0.1.pt",
            "sv-v0-forward": f"{hu_path}/lm-sv-large-forward-v0.1.pt",
            "sv-v0-backward": f"{hu_path}/lm-sv-large-backward-v0.1.pt",
            # Tamil
            "ta-forward": f"{hu_path}/lm-ta-opus-large-forward-v0.1.pt",
            "ta-backward": f"{hu_path}/lm-ta-opus-large-backward-v0.1.pt",
            # Spanish clinical
            "es-clinical-forward": f"{hu_path}/es-clinical-forward.pt",
            "es-clinical-backward": f"{hu_path}/es-clinical-backward.pt",
            # CLEF HIPE Shared task
            "de-impresso-hipe-v1-forward": f"{clef_hipe_path}/de-hipe-flair-v1-forward/best-lm.pt",
            "de-impresso-hipe-v1-backward": f"{clef_hipe_path}/de-hipe-flair-v1-backward/best-lm.pt",
            "en-impresso-hipe-v1-forward": f"{clef_hipe_path}/en-flair-v1-forward/best-lm.pt",
            "en-impresso-hipe-v1-backward": f"{clef_hipe_path}/en-flair-v1-backward/best-lm.pt",
            "fr-impresso-hipe-v1-forward": f"{clef_hipe_path}/fr-hipe-flair-v1-forward/best-lm.pt",
            "fr-impresso-hipe-v1-backward": f"{clef_hipe_path}/fr-hipe-flair-v1-backward/best-lm.pt",
            # Amharic
            "am-forward": f"{am_path}/best-lm.pt",
            # Ukrainian
            "uk-forward": "https://huggingface.co/dchaplinsky/flair-uk-forward/resolve/main/best-lm.pt",
            "uk-backward": "https://huggingface.co/dchaplinsky/flair-uk-backward/resolve/main/best-lm.pt",
        }

        if isinstance(model, str):
            # load model if in pretrained model map
            if model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]

                # Fix for CLEF HIPE models (avoid overwriting best-lm.pt in cache_dir)
                if "impresso-hipe" in model.lower():
                    cache_dir = cache_dir / model.lower()
                    # CLEF HIPE models are lowercased
                    self.is_lower = True
                model = cached_path(base_path, cache_dir=cache_dir)

            elif replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[replace_with_language_code(model)]
                model = cached_path(base_path, cache_dir=cache_dir)

            elif not Path(model).exists():
                raise ValueError(f'The given model "{model}" is not available or is not a valid path.')

        from flair.models import LanguageModel

        if isinstance(model, LanguageModel):
            self.lm: LanguageModel = model
            self.name = f"Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}"
        else:
            self.lm = LanguageModel.load_language_model(model, has_decoder=has_decoder)
            self.name = str(model)

        if name is not None:
            self.name = name

        # embeddings are static if we don't do finetuning
        self.fine_tune = fine_tune
        self.static_embeddings = not fine_tune

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.with_whitespace: bool = with_whitespace
        self.tokenized_lm: bool = tokenized_lm
        self.chars_per_chunk: int = chars_per_chunk

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence("hello")
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0][0].get_embedding())

        # set to eval mode
        self.eval()

    def train(self, mode=True):
        # unless fine-tuning is set, do not set language model to train() in order to disallow language model dropout
        super().train(self.fine_tune and mode)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:
            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = (
                [sentence.to_tokenized_string() for sentence in sentences]
                if self.tokenized_lm
                else [sentence.to_plain_string() for sentence in sentences]
            )

            if self.is_lower:
                text_sentences = [sentence.lower() for sentence in text_sentences]

            start_marker = self.lm.document_delimiter if "document_delimiter" in self.lm.__dict__ else "\n"
            end_marker = " "

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(
                text_sentences, start_marker, end_marker, self.chars_per_chunk
            )

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string() if self.tokenized_lm else sentence.to_plain_string()

                offset_forward: int = len(start_marker)
                offset_backward: int = len(sentence_text) + len(start_marker)

                for token in sentence.tokens:
                    offset_forward += len(token.text)
                    if self.is_forward_lm:
                        offset_with_whitespace = offset_forward
                        offset_without_whitespace = offset_forward - 1
                    else:
                        offset_with_whitespace = offset_backward
                        offset_without_whitespace = offset_backward - 1

                    # offset mode that extracts at whitespace after last character
                    if self.with_whitespace:
                        embedding = all_hidden_states_in_lm[offset_with_whitespace, i, :]
                    # offset mode that extracts at last character
                    else:
                        embedding = all_hidden_states_in_lm[offset_without_whitespace, i, :]

                    if self.tokenized_lm or token.whitespace_after > 0:
                        offset_forward += 1
                        offset_backward -= 1

                    offset_backward -= len(token.text)

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences

    def __str__(self) -> str:
        return self.name

    def to_params(self):
        return {
            "fine_tune": self.fine_tune,
            "chars_per_chunk": self.chars_per_chunk,
            "is_lower": self.is_lower,
            "tokenized_lm": self.tokenized_lm,
            "model_params": {
                "dictionary": self.lm.dictionary,
                "is_forward_lm": self.lm.is_forward_lm,
                "hidden_size": self.lm.hidden_size,
                "nlayers": self.lm.nlayers,
                "embedding_size": self.lm.embedding_size,
                "nout": self.lm.nout,
                "document_delimiter": self.lm.document_delimiter,
                "dropout": self.lm.dropout,
                "has_decoder": self.lm.decoder is not None,
            },
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params):
        model_params = params.pop("model_params")
        from flair.models import LanguageModel

        lm = LanguageModel(**model_params)
        return cls(lm, **params)

    def __setstate__(self, d: dict[str, Any]):
        # make compatible with old models
        d.setdefault("fine_tune", False)
        d.setdefault("chars_per_chunk", 512)
        d.setdefault("with_whitespace", True)
        d.setdefault("tokenized_lm", True)
        d.setdefault("is_lower", False)
        d.setdefault("field", None)

        super().__setstate__(d)


@register_embeddings
class PooledFlairEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        contextual_embeddings: Union[str, FlairEmbeddings],
        pooling: str = "min",
        only_capitalized: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        # use the character language model embeddings as basis
        if isinstance(contextual_embeddings, str):
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(contextual_embeddings, **kwargs)
        else:
            self.context_embeddings = contextual_embeddings

        # length is twice the original character LM embedding length
        self.__embedding_length = self.context_embeddings.embedding_length * 2
        self.name = self.context_embeddings.name + "-context"

        # these fields are for the embedding memory
        self.word_embeddings: dict[str, torch.Tensor] = {}
        self.word_count: dict[str, int] = {}

        # whether to add only capitalized words to memory (faster runtime and lower memory consumption)
        self.only_capitalized = only_capitalized

        # we re-compute embeddings dynamically at each epoch
        self.static_embeddings = False

        # set the memory method
        self.pooling = pooling

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            # memory is wiped each time we do a training run
            log.info("train mode resetting embeddings")
            self.word_embeddings = {}
            self.word_count = {}

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        self.context_embeddings.embed(sentences)

        # if we keep a pooling, it needs to be updated continuously
        for sentence in sentences:
            for token in sentence.tokens:
                # update embedding
                local_embedding = token._embeddings[self.context_embeddings.name].cpu()

                # check token.text is empty or not
                if token.text and (token.text[0].isupper() or not self.only_capitalized):
                    if token.text not in self.word_embeddings:
                        self.word_embeddings[token.text] = local_embedding
                        self.word_count[token.text] = 1
                    else:
                        # set aggregation operation
                        if self.pooling == "mean":
                            aggregated_embedding = torch.add(self.word_embeddings[token.text], local_embedding)
                        elif self.pooling == "fade":
                            aggregated_embedding = torch.add(self.word_embeddings[token.text], local_embedding)
                            aggregated_embedding /= 2
                        elif self.pooling == "max":
                            aggregated_embedding = torch.max(self.word_embeddings[token.text], local_embedding)
                        elif self.pooling == "min":
                            aggregated_embedding = torch.min(self.word_embeddings[token.text], local_embedding)

                        self.word_embeddings[token.text] = aggregated_embedding
                        self.word_count[token.text] += 1

        # add embeddings after updating
        for sentence in sentences:
            for token in sentence.tokens:
                if token.text in self.word_embeddings:
                    base = (
                        self.word_embeddings[token.text] / self.word_count[token.text]
                        if self.pooling == "mean"
                        else self.word_embeddings[token.text]
                    )
                else:
                    base = token._embeddings[self.context_embeddings.name]

                token.set_embedding(self.name, base)

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def get_names(self) -> list[str]:
        return [self.name, self.context_embeddings.name]

    def __setstate__(self, d: dict[str, Any]):
        super().__setstate__(d)

        if flair.device.type != "cpu":
            for key in self.word_embeddings:
                self.word_embeddings[key] = self.word_embeddings[key].cpu()

    @classmethod
    def from_params(cls, params):
        return cls(contextual_embeddings=load_embeddings(params.pop("contextual_embeddings")), **params)

    def to_params(self):
        return {
            "pooling": self.pooling,
            "only_capitalized": self.only_capitalized,
            "contextual_embeddings": self.context_embeddings.save_embeddings(use_state_dict=False),
        }


@register_embeddings
@deprecated(
    reason="The FastTextEmbeddings are no longer supported and will be removed at version 0.16.0", version="0.14.0"
)
class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings with oov functionality."""

    def __init__(
        self, embeddings: str, use_local: bool = True, field: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """Initializes fasttext word embeddings.

        Constructor downloads required embedding file and stores in cache if use_local is False.

        Args:
            embeddings: path to your embeddings '.bin' file
            use_local: set this to False if you are using embeddings from a remote source
            field: if given, the word-embeddings embed the data for the specific label-type instead of the plain text.
            name: The name of the embeddings.
        """
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        cache_dir = Path("embeddings")

        if use_local:
            embeddings_path = Path(embeddings)
            if not embeddings_path.exists():
                raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings_path = cached_path(f"{embeddings}", cache_dir=cache_dir)

        self.embeddings = embeddings_path

        self.name: str = str(embeddings_path)

        self.static_embeddings = True

        FastTextKeyedVectors, load_facebook_vectors = lazy_import(
            "word-embeddings", "gensim.models.fasttext", "FastTextKeyedVectors", "load_facebook_vectors"
        )

        if embeddings_path.suffix == ".bin":
            self.precomputed_word_embeddings = load_facebook_vectors(str(embeddings_path))
        else:
            self.precomputed_word_embeddings = FastTextKeyedVectors.load(str(embeddings_path))

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @instance_lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) -> torch.Tensor:
        word_embedding = self.precomputed_word_embeddings.get_vector(word)

        word_embedding = torch.tensor(word_embedding.tolist(), device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        for sentence in sentences:
            for token in sentence.tokens:
                word = token.text if self.field is None else token.get_label(self.field).value

                word_embedding = self.get_cached_vec(word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self) -> str:
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"

    @classmethod
    def from_params(cls, params):
        fasttext_binary = params.pop("fasttext_binary")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_path = temp_path / "fasttext.model"
            out_path.write_bytes(fasttext_binary)
            return cls(**params, embeddings=str(out_path), use_local=True)

    def to_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_path = temp_path / "fasttext.model"
            self.precomputed_word_embeddings.save(str(out_path), separately=[])
            return {"name": self.name, "field": self.field, "fasttext_binary": out_path.read_bytes()}


@register_embeddings
class OneHotEmbeddings(TokenEmbeddings):
    """One-hot encoded embeddings."""

    def __init__(
        self,
        vocab_dictionary: Dictionary,
        field: str = "text",
        embedding_length: int = 300,
        stable: bool = False,
    ) -> None:
        """Initializes one-hot encoded word embeddings and a trainable embedding layer.

        Args:
            vocab_dictionary: the vocabulary that will be encoded
            field: by default, the 'text' of tokens is embedded, but you can also embed tags such as 'pos'
            embedding_length: dimensionality of the trainable embedding layer
            stable: if True, use the stable embeddings as described in https://arxiv.org/abs/2110.02861
        """
        super().__init__()
        self.name = f"one-hot-{field}"
        self.static_embeddings = False
        self.field = field
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        self.__embedding_length = embedding_length
        self.vocab_dictionary = vocab_dictionary

        log.info(self.vocab_dictionary.idx2item)
        log.info(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = nn.Embedding(len(self.vocab_dictionary), self.__embedding_length)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        if stable:
            self.layer_norm: Optional[nn.LayerNorm] = nn.LayerNorm(embedding_length)
        else:
            self.layer_norm = None

        self.to(flair.device)
        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        tokens = [t for sentence in sentences for t in sentence.tokens]

        if self.field == "text":
            one_hot_sentences = [self.vocab_dictionary.get_idx_for_item(t.text) for t in tokens]
        else:
            one_hot_sentences = [self.vocab_dictionary.get_idx_for_item(t.get_label(self.field).value) for t in tokens]

        one_hot_sentences_tensor = torch.tensor(one_hot_sentences, dtype=torch.long).to(flair.device)

        embedded = self.embedding_layer.forward(one_hot_sentences_tensor)
        if self.layer_norm is not None:
            embedded = self.layer_norm(embedded)

        for emb, token in zip(embedded, tokens):
            token.set_embedding(self.name, emb)

        return sentences

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_corpus(cls, corpus: Corpus, field: str = "text", min_freq: int = 3, **kwargs):
        vocab_dictionary = Dictionary()
        assert corpus.train is not None
        tokens = [s.tokens for s in _iter_dataset(corpus.train)]
        tokens = [token for sublist in tokens for token in sublist]

        if field == "text":
            most_common = Counter([t.text for t in tokens]).most_common()
        else:
            most_common = Counter([t.get_label(field).value for t in tokens]).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        for token in tokens:
            vocab_dictionary.add_item(token)

        return cls(vocab_dictionary, field=field, **kwargs)

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    def to_params(self):
        return {
            "vocab_dictionary": self.vocab_dictionary,
            "field": self.field,
            "embedding_length": self.__embedding_length,
            "stable": self.layer_norm is not None,
        }


@register_embeddings
class HashEmbeddings(TokenEmbeddings):
    """Standard embeddings with Hashing Trick."""

    def __init__(self, num_embeddings: int = 1000, embedding_length: int = 300, hash_method="md5") -> None:
        super().__init__()
        self.name = "hash"
        self.static_embeddings = False
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        self.__num_embeddings = num_embeddings
        self.__embedding_length = embedding_length

        self.__hash_method = hash_method

        # model architecture
        self.embedding_layer = torch.nn.Embedding(self.__num_embeddings, self.__embedding_length)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

        self.to(flair.device)
        self.eval()

    @property
    def num_embeddings(self) -> int:
        return self.__num_embeddings

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]):
        def get_idx_for_item(text):
            hash_function = hashlib.new(self.__hash_method)
            hash_function.update(bytes(str(text), "utf-8"))
            return int(hash_function.hexdigest(), 16) % self.__num_embeddings

        context_idxs = [get_idx_for_item(t.text) for sentence in sentences for t in sentence.tokens]

        hash_sentences = torch.tensor(context_idxs, dtype=torch.long).to(flair.device)

        embedded = self.embedding_layer.forward(hash_sentences)

        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    def to_params(self):
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_length": self.embedding_length,
            "hash_method": self.__hash_method,
        }


@register_embeddings
class MuseCrosslingualEmbeddings(TokenEmbeddings):
    def __init__(
        self,
    ) -> None:
        self.name: str = "muse-crosslingual"
        self.static_embeddings = True
        self.__embedding_length: int = 300
        self.language_embeddings: dict[str, Any] = {}
        (KeyedVectors,) = lazy_import("word-embeddings", "gensim.models", "KeyedVectors")
        self.kv = KeyedVectors
        super().__init__()
        self.eval()

    @instance_lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, language_code: str, word: str) -> torch.Tensor:
        current_embedding_model = self.language_embeddings[language_code]
        if word in current_embedding_model:
            word_embedding = current_embedding_model[word]
        elif word.lower() in current_embedding_model:
            word_embedding = current_embedding_model[word.lower()]
        elif re.sub(r"\d", "#", word.lower()) in current_embedding_model:
            word_embedding = current_embedding_model[re.sub(r"\d", "#", word.lower())]
        elif re.sub(r"\d", "0", word.lower()) in current_embedding_model:
            word_embedding = current_embedding_model[re.sub(r"\d", "0", word.lower())]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype="float")
        word_embedding = torch.tensor(word_embedding, device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        for _i, sentence in enumerate(sentences):
            language_code = sentence.get_language_code()
            supported = [
                "en",
                "de",
                "bg",
                "ca",
                "hr",
                "cs",
                "da",
                "nl",
                "et",
                "fi",
                "fr",
                "el",
                "he",
                "hu",
                "id",
                "it",
                "mk",
                "no",
                # "pl",
                "pt",
                "ro",
                "ru",
                "sk",
            ]
            if language_code not in supported:
                language_code = "en"

            if language_code not in self.language_embeddings:
                log.info(f"Loading up MUSE embeddings for '{language_code}'!")
                # download if necessary
                hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/muse"
                cache_dir = Path("embeddings") / "MUSE"
                cached_path(
                    f"{hu_path}/muse.{language_code}.vec.gensim.vectors.npy",
                    cache_dir=cache_dir,
                )
                embeddings_file = cached_path(f"{hu_path}/muse.{language_code}.vec.gensim", cache_dir=cache_dir)

                # load the model
                self.language_embeddings[language_code] = self.kv.load(str(embeddings_file))

            for token, _token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = self.get_cached_vec(language_code=language_code, word=token.text)

                token.set_embedding(self.name, word_embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_params(cls, params):
        return cls()

    def to_params(self):
        return {}


@register_embeddings
class BytePairEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        language: Optional[str] = None,
        dim: int = 50,
        syllables: int = 100000,
        cache_dir=None,
        model_file_path: Optional[Path] = None,
        embedding_file_path: Optional[Path] = None,
        name: Optional[str] = None,
        force_cpu: bool = True,
        field: Optional[str] = None,
        preprocess: bool = True,
        **kwargs,
    ) -> None:
        """Initializes BP embeddings.

        Constructor downloads required files if not there.
        """
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        if not cache_dir:
            cache_dir = flair.cache_root / "embeddings"

        if model_file_path is not None and embedding_file_path is None:
            self.spm = SentencePieceProcessor()
            self.spm.Load(str(model_file_path))
            vectors = np.zeros((self.spm.vocab_size() + 1, dim))
            if name is not None:
                self.name = name
            else:
                raise ValueError(
                    "When only providing a SentencePieceProcessor, you need to specify a name for the BytePairEmbeddings"
                )
        else:
            if not language and model_file_path is None:
                raise ValueError("Need to specify model_file_path if no language is give in BytePairEmbeddings")
            (BPEmb,) = lazy_import("word-embeddings", "bpemb", "BPEmb")

            if language:
                self.name: str = f"bpe-{language}-{syllables}-{dim}"
                embedder = BPEmb(
                    lang=language,
                    vs=syllables,
                    dim=dim,
                    cache_dir=cache_dir,
                    model_file=model_file_path,
                    emb_file=embedding_file_path,
                    **kwargs,
                )
            else:
                if model_file_path is None:
                    raise ValueError("Need to specify model_file_path if no language is give in BytePairEmbeddings")
                embedder = BPEmb(
                    lang=language,
                    vs=syllables,
                    dim=dim,
                    cache_dir=cache_dir,
                    model_file=model_file_path,
                    emb_file=embedding_file_path,
                    **kwargs,
                )
            self.spm = embedder.spm
            vectors = np.vstack(
                (
                    embedder.vectors,
                    np.zeros(embedder.dim, dtype=embedder.vectors.dtype),
                )
            )
            dim = embedder.dim
            syllables = embedder.vs

            if not language:
                self.name = f"bpe-custom-{syllables}-{dim}"
        if name is not None:
            self.name = name
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vectors), freeze=True)
        self.force_cpu = force_cpu
        self.static_embeddings = True
        self.field = field
        self.do_preproc = preprocess

        self.__embedding_length: int = dim * 2
        self.eval()
        self.to(flair.device)

    def _preprocess(self, text: str) -> str:
        return re.sub(r"\d", "0", text)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: list[Sentence]) -> list[Sentence]:
        tokens = [token for sentence in sentences for token in sentence.tokens]

        word_indices: list[list[int]] = []
        for token in tokens:
            word = token.text if self.field is None else token.get_label(self.field).value

            if word.strip() == "":
                ids = [self.spm.vocab_size(), self.embedder.spm.vocab_size()]
            else:
                if self.do_preproc:
                    word = self._preprocess(word)
                ids = self.spm.EncodeAsIds(word.lower())
                ids = [ids[0], ids[-1]]
            word_indices.append(ids)

        index_tensor = torch.tensor(word_indices, dtype=torch.long, device=self.device)
        embeddings = self.embedding(index_tensor)
        embeddings = embeddings.reshape((-1, self.embedding_length))
        if self.force_cpu:
            embeddings = embeddings.to(flair.device)

        for emb, token in zip(embeddings, tokens):
            token.set_embedding(self.name, emb)

        return sentences

    def __str__(self) -> str:
        return self.name

    def extra_repr(self):
        return f"model={self.name}"

    @classmethod
    def from_params(cls, params):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file_path = temp_path / "model.spm"
            model_file_path.write_bytes(params["spm_model_binary"])

            if "word2vec_binary" in params:
                embedding_file_path = temp_path / "word2vec.bin"
                embedding_file_path.write_bytes(params["word2vec_binary"])
                dim = None
            else:
                embedding_file_path = None
                dim = params["dim"]
            return cls(
                name=params["name"],
                dim=dim,
                model_file_path=model_file_path,
                embedding_file_path=embedding_file_path,
                field=params.get("field"),
                preprocess=params.get("preprocess", True),
            )

    def to_params(self):
        return {
            "name": self.name,
            "spm_model_binary": self.spm.serialized_model_proto(),
            "dim": self.embedding_length // 2,
            "field": self.field,
            "preprocess": self.do_preproc,
        }

    def to(self, device):
        if self.force_cpu:
            device = torch.device("cpu")
        self.device = device
        super().to(device)

    def _apply(self, fn):
        if fn.__name__ == "convert" and self.force_cpu:
            # this is required to force the module on the cpu,
            # if a parent module is put to gpu, the _apply is called to each sub_module
            # self.to(..) actually sets the device properly
            if not hasattr(self, "device"):
                self.to(flair.device)
            return
        super()._apply(fn)

    def state_dict(self, *args, **kwargs):
        # when loading the old versions from pickle, the embeddings might not be added as pytorch module.
        # we do this delayed, when the weights are collected (e.g. for saving), as doing this earlier might
        # lead to issues while loading (trying to load weights that weren't stored as python weights and therefore
        # not finding them)
        if list(self.modules()) == [self]:
            self.embedding = self.embedding
        return super().state_dict(*args, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if not state_dict:
            # old embeddings do not have a torch-embedding and therefore do not store the weights in the saved torch state_dict
            # however they are already initialized rightfully, so we just set the state dict from our current state dict
            for k, v in self.state_dict(prefix=prefix).items():
                state_dict[k] = v
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


@register_embeddings
class NILCEmbeddings(WordEmbeddings):
    def __init__(self, embeddings: str, model: str = "skip", size: int = 100) -> None:
        """Initializes portuguese classic word embeddings trained by NILC Lab.

        See: http://www.nilc.icmc.usp.br/embeddings
        Constructor downloads required files if not there.

        Args:
            embeddings: one of: 'fasttext', 'glove', 'wang2vec' or 'word2vec'
            model: one of: 'skip' or 'cbow'. This is not applicable to glove.
            size: one of: 50, 100, 300, 600 or 1000.
        """
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        base_path = "http://143.107.183.175:22980/download.php?file=embeddings/"

        cache_dir = Path("embeddings") / ("nilc-" + embeddings.lower())

        # GLOVE embeddings
        if embeddings.lower() == "glove":
            cached_path(f"{base_path}{embeddings}/{embeddings}_s{size}.zip", cache_dir=cache_dir)
            embeddings_path = f"{base_path}{embeddings}/{embeddings}_s{size}.zip"

        elif embeddings.lower() in ["fasttext", "wang2vec", "word2vec"]:
            cached_path(f"{base_path}{embeddings}/{model}_s{size}.zip", cache_dir=cache_dir)
            embeddings_path = f"{base_path}{embeddings}/{model}_s{size}.zip"

        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings_path = embeddings

        log.info("Reading embeddings from %s", embeddings_path)
        super().__init__(
            embeddings=str(extract_single_zip_file(embeddings_path, cache_dir=cache_dir)), name="NILC-" + embeddings
        )

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "WordEmbeddings":
        #  no need to recreate as NILCEmbeddings
        return WordEmbeddings(embeddings=None, **params)


def replace_with_language_code(string: str):
    string = string.replace("arabic-", "ar-")
    string = string.replace("basque-", "eu-")
    string = string.replace("bulgarian-", "bg-")
    string = string.replace("croatian-", "hr-")
    string = string.replace("czech-", "cs-")
    string = string.replace("danish-", "da-")
    string = string.replace("dutch-", "nl-")
    string = string.replace("farsi-", "fa-")
    string = string.replace("persian-", "fa-")
    string = string.replace("finnish-", "fi-")
    string = string.replace("french-", "fr-")
    string = string.replace("german-", "de-")
    string = string.replace("hebrew-", "he-")
    string = string.replace("hindi-", "hi-")
    string = string.replace("indonesian-", "id-")
    string = string.replace("italian-", "it-")
    string = string.replace("japanese-", "ja-")
    string = string.replace("norwegian-", "no")
    string = string.replace("polish-", "pl-")
    string = string.replace("portuguese-", "pt-")
    string = string.replace("slovenian-", "sl-")
    string = string.replace("spanish-", "es-")
    string = string.replace("swedish-", "sv-")
    return string


__all__ = [
    "TransformerWordEmbeddings",
    "StackedEmbeddings",
    "WordEmbeddings",
    "CharacterEmbeddings",
    "FlairEmbeddings",
    "PooledFlairEmbeddings",
    "FastTextEmbeddings",
    "OneHotEmbeddings",
    "HashEmbeddings",
    "MuseCrosslingualEmbeddings",
    "BytePairEmbeddings",
    "NILCEmbeddings",
]
