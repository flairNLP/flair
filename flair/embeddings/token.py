import hashlib
import logging
import glob, os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from io import open

import gensim
import numpy as np
import torch
from bpemb import BPEmb
from gensim.models import KeyedVectors
from mypy.plugins import ctypes
from torch import nn

import flair
from flair.data import Corpus, Dictionary, Sentence, Token, _iter_dataset
from flair.embeddings.base import Embeddings, TransformerEmbedding
from flair.file_utils import cached_path, instance_lru_cache, open_inside_zip

log = logging.getLogger("flair")


class TokenEmbeddings(Embeddings[Sentence]):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    def embedding_type(self) -> str:
        return "word-level"

    def _everything_embedded(self, data_points: Sequence[Sentence]) -> bool:
        for sentence in data_points:
            for token in sentence.tokens:
                if self.name not in token._embeddings.keys():
                    return False
        return True


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True):
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

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        names = []
        for embedding in self.embeddings:
            names.extend(embedding.get_names())
        return names

    def get_named_embeddings_dict(self) -> Dict:

        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict


class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(
            self,
            embeddings: str,
            field: str = None,
            fine_tune: bool = False,
            force_cpu: bool = True,
            stable: bool = False,
    ):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        set stable=True to use the stable embeddings as described in https://arxiv.org/abs/2110.02861
        """
        self.embeddings = embeddings

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        if fine_tune and force_cpu and flair.device.type != "cpu":
            raise ValueError("Cannot train WordEmbeddings on cpu if the model is trained on gpu, set force_cpu=False")

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/token"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{hu_path}/glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/glove.gensim", cache_dir=cache_dir)

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(f"{hu_path}/turian.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/turian", cache_dir=cache_dir)

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(f"{hu_path}/extvec.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/extvec.gensim", cache_dir=cache_dir)

        # pubmed embeddings
        elif embeddings.lower() == "pubmed" or embeddings.lower() == "en-pubmed":
            cached_path(
                f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() in ["news", "en-news", "en"]:
            cached_path(f"{hu_path}/en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/en-fasttext-news-300d-1M", cache_dir=cache_dir)

        # twitter embeddings
        elif embeddings.lower() in ["twitter", "en-twitter"]:
            cached_path(f"{hu_path}/twitter.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{hu_path}/twitter.gensim", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(
                f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings_path = cached_path(
                f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings_path = Path(embeddings)

        self.name: str = str(embeddings_path)
        self.static_embeddings = not fine_tune
        self.fine_tune = fine_tune
        self.force_cpu = force_cpu
        self.field = field
        self.stable = stable
        super().__init__()

        if embeddings_path.suffix == ".bin":
            precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings_path), binary=True
            )
        else:
            precomputed_word_embeddings = gensim.models.KeyedVectors.load(str(embeddings_path))

        self.__embedding_length: int = precomputed_word_embeddings.vector_size

        vectors = np.row_stack(
            (
                precomputed_word_embeddings.vectors,
                np.zeros(self.__embedding_length, dtype="float"),
            )
        )
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vectors), freeze=not fine_tune)

        try:
            # gensim version 4
            self.vocab = precomputed_word_embeddings.key_to_index
        except AttributeError:
            # gensim version 3
            self.vocab = {k: v.index for k, v in precomputed_word_embeddings.vocab.items()}

        if stable:
            self.layer_norm: Optional[nn.LayerNorm] = nn.LayerNorm(
                self.__embedding_length, elementwise_affine=fine_tune
            )
        else:
            self.layer_norm = None

        self.device = None
        self.to(flair.device)

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

    def get_vec(self, word: str) -> torch.Tensor:
        word_embedding = self.vectors[self.get_cached_token_index(word)]

        word_embedding = torch.tensor(word_embedding.tolist(), device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        tokens = [token for sentence in sentences for token in sentence.tokens]

        word_indices: List[int] = []
        for token in tokens:
            if "field" not in self.__dict__ or self.field is None:
                word = token.text
            else:
                word = token.get_tag(self.field).value
            word_indices.append(self.get_cached_token_index(word))

        embeddings = self.embedding(torch.tensor(word_indices, dtype=torch.long, device=self.device))
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        if self.force_cpu:
            embeddings = embeddings.to(flair.device)

        for emb, token in zip(embeddings, tokens):
            token.set_embedding(self.name, emb)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        # fix serialized models
        if "embeddings" not in self.__dict__:
            self.embeddings = self.name

        return f"'{self.embeddings}'"

    def train(self, mode=True):
        if not self.fine_tune:
            pass
        else:
            super(WordEmbeddings, self).train(mode)

    def to(self, device):
        if self.force_cpu:
            device = torch.device("cpu")
        self.device = device
        super(WordEmbeddings, self).to(device)

    def _apply(self, fn):
        if fn.__name__ == "convert" and self.force_cpu:
            # this is required to force the module on the cpu,
            # if a parent module is put to gpu, the _apply is called to each sub_module
            # self.to(..) actually sets the device properly
            if not hasattr(self, "device"):
                self.to(flair.device)
            return
        super(WordEmbeddings, self)._apply(fn)

    def __getattribute__(self, item):
        # this ignores the get_cached_vec method when loading older versions
        # it is needed for compatibility reasons
        if "get_cached_vec" == item:
            return None
        return super().__getattribute__(item)

    def __setstate__(self, state):
        if "get_cached_vec" in state:
            del state["get_cached_vec"]
        if "force_cpu" not in state:
            state["force_cpu"] = True
        if "fine_tune" not in state:
            state["fine_tune"] = False
        if "precomputed_word_embeddings" in state:
            precomputed_word_embeddings: KeyedVectors = state.pop("precomputed_word_embeddings")
            vectors = np.row_stack(
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


class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
            self,
            path_to_char_dict: str = None,
            char_embedding_dim: int = 25,
            hidden_size_char: int = 25,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
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

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

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

            packed = torch.nn.utils.rnn.pack_padded_sequence(character_embeddings, chars2_length)  # type: ignore

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)),
                dtype=torch.float,
                device=flair.device,
            )
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


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
    ):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward',
                etc (see https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)
                depending on which character language model is desired.
        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows
                down training and often leads to overfitting, so use with caution.
        :param chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster
                but requires more memory. Lower means slower but less memory.
        :param with_whitespace: If True, use hidden state after whitespace after word. If False, use hidden
                 state at last character of word.
        :param tokenized_lm: Whether this lm is tokenized. Default is True, but for LMs trained over unprocessed text
                False might be better.
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
        }

        if type(model) == str:

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

        if type(model) == LanguageModel:
            self.lm: LanguageModel = model
            self.name = f"Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}"
        else:
            self.lm = LanguageModel.load_language_model(model)
            self.name = str(model)

        # embeddings are static if we don't do finetuning
        self.fine_tune = fine_tune
        self.static_embeddings = not fine_tune

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.with_whitespace: bool = with_whitespace
        self.tokenized_lm: bool = tokenized_lm
        self.chars_per_chunk: int = chars_per_chunk

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0][0].get_embedding())

        # set to eval mode
        self.eval()

    def train(self, mode=True):

        # make compatible with serialized models (TODO: remove)
        if "fine_tune" not in self.__dict__:
            self.fine_tune = False
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        # unless fine-tuning is set, do not set language model to train() in order to disallow language model dropout
        if not self.fine_tune:
            pass
        else:
            super(FlairEmbeddings, self).train(mode)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # make compatible with serialized models (TODO: remove)
        if "with_whitespace" not in self.__dict__:
            self.with_whitespace = True
        if "tokenized_lm" not in self.__dict__:
            self.tokenized_lm = True
        if "is_lower" not in self.__dict__:
            self.is_lower = False

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

                    if self.tokenized_lm or token.whitespace_after:
                        offset_forward += 1
                        offset_backward -= 1

                    offset_backward -= len(token.text)

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            contextual_embeddings: Union[str, FlairEmbeddings],
            pooling: str = "min",
            only_capitalized: bool = False,
            **kwargs,
    ):

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
        self.word_embeddings: Dict[str, torch.Tensor] = {}
        self.word_count: Dict[str, int] = {}

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
            print("train mode resetting embeddings")
            self.word_embeddings = {}
            self.word_count = {}

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.context_embeddings.embed(sentences)

        # if we keep a pooling, it needs to be updated continuously
        for sentence in sentences:
            for token in sentence.tokens:

                # update embedding
                local_embedding = token._embeddings[self.context_embeddings.name].cpu()

                # check token.text is empty or not
                if token.text:
                    if token.text[0].isupper() or not self.only_capitalized:

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

    def embedding_length(self) -> int:
        return self.__embedding_length

    def get_names(self) -> List[str]:
        return [self.name, self.context_embeddings.name]

    def __setstate__(self, d):
        self.__dict__ = d

        if flair.device != "cpu":
            for key in self.word_embeddings:
                self.word_embeddings[key] = self.word_embeddings[key].cpu()


class GazetteerEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            path_to_gazetteers: str,
            label_dict: dict,
            full_mathing: bool = True,
            partial_matching: bool = True
    ):
        super().__init__()
        self.name = "Gazetteer"
        self.gazetteer_path = path_to_gazetteers
        self.labels = label_dict
        self.matching_methods = []
        if full_mathing:
            self.matching_methods.append('full_match')
        if partial_matching:
            self.matching_methods.append('partial_match')

        self.gazetteer_file_dict_list = self._get_gazetteers()
        self.gazetteers_dicts = self._process_gazetteers()
        self.feature_list = self._set_feature_list()
        self.__embedding_length = len(self.feature_list)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _set_feature_list(self):
        temp_1 = []
        temp_2 = []
        for key in self.labels.keys():
            if not 'O' == key:
                if 'partial_match' in self.matching_methods:
                    temp_1.append(key)
                if 'full_match' in self.matching_methods:
                    if 'B-' in key:
                        temp_2.append(key[2:])
        return ['O', *temp_1, *temp_2]

    def _get_gazetteers(self):
        gazetteer_files = []
        for key in self.labels.keys():
            dict_with_label_and_files = {}
            if 'B-' in key:
                key = key[2:]
                pattern = f'.*-?{key}-.*[.]txt'
                temp_list = []
                temp_list.extend([f for f in os.listdir(self.gazetteer_path + '/') if re.match(pattern, f)])
                dict_with_label_and_files[key] = temp_list
                gazetteer_files.append(dict_with_label_and_files)
        return gazetteer_files

    def _filter_gazetteer_line(self, line_list):
        line_list_filtered = []
        for w in line_list:
            w = w.rstrip("\n")
            if w.isalpha():
                line_list_filtered.append(w)
            elif len(w) > 1:
                line_list_filtered.append(w)
        return line_list_filtered

    def _check_for_ner_tag_already_in_dict(self, matching_list, ner_tag):
        gazetteer_tag_dict_list = {}
        hash_dict_index = None
        for elem in matching_list:
            if f'B-{ner_tag}' in elem.keys() or ner_tag in elem.keys():
                gazetteer_tag_dict_list = elem
                hash_dict_index = matching_list.index(elem)
        return gazetteer_tag_dict_list, hash_dict_index

    def _get_hash_of_string(self, text):
        h = hashlib.new('sha256')
        h.update(bytes(str(text), "utf-8"))
        return str(h.hexdigest())

    def _process_gazetteers(self):
        partial_matching_dict_list = []
        full_matching_dict_list = []

        for gazetteer_dict in self.gazetteer_file_dict_list:
            tag_key = list(gazetteer_dict.keys())[0]
            gazetteer_files = gazetteer_dict[tag_key]
            if len(gazetteer_files) > 0:
                for gazetteer_file in gazetteer_files:
                    gazetteer_tag_dict_list_partial, hash_dict_index_partial = \
                        self._check_for_ner_tag_already_in_dict(partial_matching_dict_list, tag_key)
                    gazetteer_tag_dict_dict_full, hash_dict_index_full = \
                        self._check_for_ner_tag_already_in_dict(full_matching_dict_list, tag_key)
                    with open(f'{self.gazetteer_path}/{gazetteer_file}', 'r', encoding='utf-8', errors='strict') as src:
                        for line in src:
                            if len(line) == 0:
                                break
                            else:
                                if 'partial_match' in self.matching_methods:
                                    line_list = re.split(" ", line)
                                    line_list_filtered = self._filter_gazetteer_line(line_list)
                                    for word in line_list_filtered:
                                        word_hash = self._get_hash_of_string(word)
                                        word_index = line_list_filtered.index(word)
                                        if word_index == 0 and len(line_list_filtered) > 1:
                                            if f'B-{tag_key}' in self.labels.keys():
                                                try:
                                                    gazetteer_tag_dict_list_partial[f'B-{tag_key}'][word_hash] = word
                                                except KeyError:
                                                    gazetteer_tag_dict_list_partial[f'B-{tag_key}'] = {}
                                                    gazetteer_tag_dict_list_partial[f'B-{tag_key}'][word_hash] = word
                                        elif word_index == len(line_list_filtered) - 1 and len(line_list_filtered) > 1:
                                            if f'E-{tag_key}' in self.labels.keys():
                                                try:
                                                    gazetteer_tag_dict_list_partial[f'E-{tag_key}'][word_hash] = word
                                                except KeyError:
                                                    gazetteer_tag_dict_list_partial[f'E-{tag_key}'] = {}
                                                    gazetteer_tag_dict_list_partial[f'E-{tag_key}'][word_hash] = word
                                        elif 0 < word_index < len(line_list_filtered) - 1:
                                            if f'I-{tag_key}' in self.labels.keys():
                                                try:
                                                    gazetteer_tag_dict_list_partial[f'I-{tag_key}'][word_hash] = word
                                                except KeyError:
                                                    gazetteer_tag_dict_list_partial[f'I-{tag_key}'] = {}
                                                    gazetteer_tag_dict_list_partial[f'I-{tag_key}'][word_hash] = word
                                        elif word_index == 0 and len(line_list_filtered) == 1:
                                            if f'S-{tag_key}' in self.labels.keys():
                                                try:
                                                    gazetteer_tag_dict_list_partial[f'S-{tag_key}'][word_hash] = word
                                                except KeyError:
                                                    gazetteer_tag_dict_list_partial[f'S-{tag_key}'] = {}
                                                    gazetteer_tag_dict_list_partial[f'S-{tag_key}'][word_hash] = word
                                if 'full_match' in self.matching_methods:
                                    line = line.rstrip("\n")
                                    if len(line) > 0:
                                        line_hash = self._get_hash_of_string(line)
                                        try:
                                            gazetteer_tag_dict_dict_full[tag_key][line_hash] = line
                                        except KeyError:
                                            gazetteer_tag_dict_dict_full[tag_key] = {}
                                            gazetteer_tag_dict_dict_full[tag_key][line_hash] = line
                        if hash_dict_index_partial is not None:
                            partial_matching_dict_list[hash_dict_index_partial] = gazetteer_tag_dict_list_partial
                        else:
                            partial_matching_dict_list.append(gazetteer_tag_dict_list_partial)
                        if hash_dict_index_full is not None:
                            full_matching_dict_list[hash_dict_index_full] = gazetteer_tag_dict_dict_full
                        else:
                            full_matching_dict_list.append(gazetteer_tag_dict_dict_full)
        return {'partial_match': partial_matching_dict_list, 'full_match': full_matching_dict_list}

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sentence in sentences:
            for token in sentence.tokens:
                token_found_in_gazetteer = False
                feature_vector = [0]*len(self.feature_list)
                if 'partial_match' in self.matching_methods:
                    token_text_hash = self._get_hash_of_string(token.text)
                    for gazetteer_dict in self.gazetteers_dicts['partial_match']:
                        for gazetteer_hash_dict in gazetteer_dict.keys():
                            try:
                                if token.text == gazetteer_dict[gazetteer_hash_dict][token_text_hash]:
                                    feature_vector[self.feature_list.index(gazetteer_hash_dict)] = 1
                                    token_found_in_gazetteer = True
                            except KeyError:
                                pass
                if 'full_match' in self.matching_methods:
                    pass
                if not token_found_in_gazetteer:
                    feature_vector[self.feature_list.index('O')] = 1
                    token.set_embedding(self.name, torch.tensor(feature_vector, dtype=torch.int, device=flair.device))
                else:
                    token.set_embedding(self.name, torch.tensor(feature_vector, dtype=torch.int, device=flair.device))
        return sentences


class TransformerWordEmbeddings(TokenEmbeddings, TransformerEmbedding):
    def __init__(
            self,
            model: str = "bert-base-uncased",
            is_document_embedding: bool = False,
            allow_long_sentences: bool = True,
            **kwargs,
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param subtoken_pooling: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param layer_mean: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        """
        TransformerEmbedding.__init__(
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


class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings with oov functionality"""

    def __init__(self, embeddings: str, use_local: bool = True, field: str = None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
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

        self.precomputed_word_embeddings: gensim.models.FastText = gensim.models.FastText.load_fasttext_format(
            str(embeddings)
        )
        print(self.precomputed_word_embeddings)

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @instance_lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) -> torch.Tensor:
        try:
            word_embedding = self.precomputed_word_embeddings.wv[word]
        except IndexError:
            word_embedding = np.zeros(self.embedding_length, dtype="float")

        word_embedding = torch.tensor(word_embedding.tolist(), device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"


class OneHotEmbeddings(TokenEmbeddings):
    """One-hot encoded embeddings."""

    def __init__(
            self,
            vocab_dictionary: Dictionary,
            field: str = "text",
            embedding_length: int = 300,
            stable: bool = False,
    ):
        """
        Initializes one-hot encoded word embeddings and a trainable embedding layer
        :param vocab_dictionary: the vocabulary that will be encoded
        :param field: by default, the 'text' of tokens is embedded, but you can also embed tags such as 'pos'
        :param embedding_length: dimensionality of the trainable embedding layer
        :param stable: set stable=True to use the stable embeddings as described in https://arxiv.org/abs/2110.02861
        """
        super().__init__()
        self.name = f"one-hot-{field}"
        self.static_embeddings = False
        self.field = field
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        self.__embedding_length = embedding_length
        self.vocab_dictionary = vocab_dictionary

        print(self.vocab_dictionary.idx2item)
        print(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = nn.Embedding(len(self.vocab_dictionary), self.__embedding_length)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        if stable:
            self.layer_norm: Optional[nn.LayerNorm] = nn.LayerNorm(embedding_length)
        else:
            self.layer_norm = None

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        tokens = [t for sentence in sentences for t in sentence.tokens]

        if self.field == "text":
            one_hot_sentences = [self.vocab_dictionary.get_idx_for_item(t.text) for t in tokens]
        else:
            one_hot_sentences = [self.vocab_dictionary.get_idx_for_item(t.get_tag(self.field).value) for t in tokens]

        one_hot_sentences_tensor = torch.tensor(one_hot_sentences, dtype=torch.long).to(flair.device)

        embedded = self.embedding_layer.forward(one_hot_sentences_tensor)
        if self.layer_norm is not None:
            embedded = self.layer_norm(embedded)

        for emb, token in zip(embedded, tokens):
            token.set_embedding(self.name, emb)

        return sentences

    def __str__(self):
        return self.name

    @classmethod
    def from_corpus(cls, corpus: Corpus, field: str = "text", min_freq: int = 3, **kwargs):
        vocab_dictionary = Dictionary()
        assert corpus.train is not None
        tokens = list(map((lambda s: s.tokens), _iter_dataset(corpus.train)))
        tokens = [token for sublist in tokens for token in sublist]

        if field == "text":
            most_common = Counter(list(map((lambda t: t.text), tokens))).most_common()
        else:
            most_common = Counter(list(map((lambda t: t.get_tag(field).value), tokens))).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        for token in tokens:
            vocab_dictionary.add_item(token)

        return cls(vocab_dictionary, field=field, **kwargs)


class HashEmbeddings(TokenEmbeddings):
    """Standard embeddings with Hashing Trick."""

    def __init__(self, num_embeddings: int = 1000, embedding_length: int = 300, hash_method="md5"):

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

    @property
    def num_embeddings(self) -> int:
        return self.__num_embeddings

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        def get_idx_for_item(text):
            hash_function = hashlib.new(self.__hash_method)
            hash_function.update(bytes(str(text), "utf-8"))
            return int(hash_function.hexdigest(), 16) % self.__num_embeddings

        context_idxs = [[get_idx_for_item(t.text) for t in sentence.tokens] for sentence in sentences]

        hash_sentences = torch.tensor(context_idxs, dtype=torch.long).to(flair.device)

        embedded = self.embedding_layer.forward(hash_sentences)

        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1

    def __str__(self):
        return self.name


class MuseCrosslingualEmbeddings(TokenEmbeddings):
    def __init__(
            self,
    ):
        self.name: str = "muse-crosslingual"
        self.static_embeddings = True
        self.__embedding_length: int = 300
        self.language_embeddings = {}
        super().__init__()

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

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

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
                self.language_embeddings[language_code] = gensim.models.KeyedVectors.load(str(embeddings_file))

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(language_code=language_code, word=word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name


# TODO: keep for backwards compatibility, but remove in future
class BPEmbSerializable(BPEmb):
    def __getstate__(self):
        state = self.__dict__.copy()
        # save the sentence piece model as binary file (not as path which may change)
        state["spm_model_binary"] = open(self.model_file, mode="rb").read()
        state["spm"] = None
        return state

    def __setstate__(self, state):
        from bpemb.util import sentencepiece_load

        model_file = self.model_tpl.format(lang=state["lang"], vs=state["vs"])
        self.__dict__ = state

        # write out the binary sentence piece model into the expected directory
        self.cache_dir: Path = flair.cache_root / "embeddings"
        if "spm_model_binary" in self.__dict__:
            # if the model was saved as binary and it is not found on disk, write to appropriate path
            if not os.path.exists(self.cache_dir / state["lang"]):
                os.makedirs(self.cache_dir / state["lang"])
            self.model_file = self.cache_dir / model_file
            with open(self.model_file, "wb") as out:
                out.write(self.__dict__["spm_model_binary"])
        else:
            # otherwise, use normal process and potentially trigger another download
            self.model_file = self._load_file(model_file)

        # once the modes if there, load it with sentence piece
        state["spm"] = sentencepiece_load(self.model_file)


class BytePairEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            language: str = None,
            dim: int = 50,
            syllables: int = 100000,
            cache_dir=None,
            model_file_path: Path = None,
            embedding_file_path: Path = None,
            **kwargs,
    ):
        """
        Initializes BP embeddings. Constructor downloads required files if not there.
        """
        self.instance_parameters = self.get_instance_parameters(locals=locals())

        if not cache_dir:
            cache_dir = flair.cache_root / "embeddings"
        if language:
            self.name: str = f"bpe-{language}-{syllables}-{dim}"
        else:
            assert (
                    model_file_path is not None and embedding_file_path is not None
            ), "Need to specify model_file_path and embedding_file_path if no language is given in BytePairEmbeddings(...)"
            dim = None  # type: ignore

        self.embedder = BPEmbSerializable(
            lang=language,
            vs=syllables,
            dim=dim,
            cache_dir=cache_dir,
            model_file=model_file_path,
            emb_file=embedding_file_path,
            **kwargs,
        )

        if not language:
            self.name = f"bpe-custom-{self.embedder.vs}-{self.embedder.dim}"
        self.static_embeddings = True

        self.__embedding_length: int = self.embedder.emb.vector_size * 2
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word.strip() == "":
                    # empty words get no embedding
                    token.set_embedding(self.name, torch.zeros(self.embedding_length, dtype=torch.float))
                else:
                    # all other words get embedded
                    embeddings = self.embedder.embed(word.lower())
                    embedding = np.concatenate((embeddings[0], embeddings[len(embeddings) - 1]))
                    token.set_embedding(self.name, torch.tensor(embedding, dtype=torch.float))

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "model={}".format(self.name)


class ELMoEmbeddings(TokenEmbeddings):
    """Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018.
    ELMo word vectors can be constructed by combining layers in different ways.
    Default is to concatene the top 3 layers in the LM."""

    def __init__(
            self,
            model: str = "original",
            options_file: str = None,
            weight_file: str = None,
            embedding_mode: str = "all",
    ):
        super().__init__()

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        try:
            import allennlp.commands.elmo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning('To use ELMoEmbeddings, please first install with "pip install allennlp==0.9.0"')
            log.warning("-" * 100)
            pass

        assert embedding_mode in ["all", "top", "average"]

        self.name = f"elmo-{model}-{embedding_mode}"
        self.static_embeddings = True

        if not options_file or not weight_file:
            # the default model for ELMo is the 'original' model, which is very large
            options_file = allennlp.commands.elmo.DEFAULT_OPTIONS_FILE
            weight_file = allennlp.commands.elmo.DEFAULT_WEIGHT_FILE
            # alternatively, a small, medium or portuguese model can be selected by passing the appropriate mode name
            if model == "small":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            if model == "medium":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            if model in ["large", "5.5B"]:
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            if model == "pt" or model == "portuguese":
                options_file = (
                    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json"
                )
                weight_file = (
                    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5"
                )
            if model == "pubmed":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

        if embedding_mode == "all":
            self.embedding_mode_fn = self.use_layers_all
        elif embedding_mode == "top":
            self.embedding_mode_fn = self.use_layers_top
        elif embedding_mode == "average":
            self.embedding_mode_fn = self.use_layers_average

            # put on Cuda if available
        from flair import device

        if re.fullmatch(r"cuda:[0-9]+", str(device)):
            cuda_device = int(str(device).split(":")[-1])
        elif str(device) == "cpu":
            cuda_device = -1
        else:
            cuda_device = 0

        self.ee = allennlp.commands.elmo.ElmoEmbedder(
            options_file=options_file, weight_file=weight_file, cuda_device=cuda_device
        )

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0][0].get_embedding())

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def use_layers_all(self, x):
        return torch.cat(x, 0)

    def use_layers_top(self, x):
        return x[-1]

    def use_layers_average(self, x):
        return torch.mean(torch.stack(x), 0)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        # ELMoEmbeddings before Release 0.5 did not set self.embedding_mode_fn
        if not getattr(self, "embedding_mode_fn", None):
            self.embedding_mode_fn = self.use_layers_all

        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])

        embeddings = self.ee.embed_batch(sentence_words)

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                elmo_embedding_layers = [
                    torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[2, token_idx, :]),
                ]
                word_embedding = self.embedding_mode_fn(elmo_embedding_layers)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name

    def __setstate__(self, state):
        self.__dict__ = state

        if re.fullmatch(r"cuda:[0-9]+", str(flair.device)):
            cuda_device = int(str(flair.device).split(":")[-1])
        elif str(flair.device) == "cpu":
            cuda_device = -1
        else:
            cuda_device = 0

        self.ee.cuda_device = cuda_device

        self.ee.elmo_bilm.to(device=flair.device)
        self.ee.elmo_bilm._elmo_lstm._states = tuple(
            [state.to(flair.device) for state in self.ee.elmo_bilm._elmo_lstm._states]
        )


class NILCEmbeddings(WordEmbeddings):
    def __init__(self, embeddings: str, model: str = "skip", size: int = 100):
        """
        Initializes portuguese classic word embeddings trained by NILC Lab (http://www.nilc.icmc.usp.br/embeddings).
        Constructor downloads required files if not there.
        :param embeddings: one of: 'fasttext', 'glove', 'wang2vec' or 'word2vec'
        :param model: one of: 'skip' or 'cbow'. This is not applicable to glove.
        :param size: one of: 50, 100, 300, 600 or 1000.
        """

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        base_path = "http://143.107.183.175:22980/download.php?file=embeddings/"

        cache_dir = Path("embeddings") / embeddings.lower()

        # GLOVE embeddings
        if embeddings.lower() == "glove":
            cached_path(f"{base_path}{embeddings}/{embeddings}_s{size}.zip", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{base_path}{embeddings}/{embeddings}_s{size}.zip", cache_dir=cache_dir)

        elif embeddings.lower() in ["fasttext", "wang2vec", "word2vec"]:
            cached_path(f"{base_path}{embeddings}/{model}_s{size}.zip", cache_dir=cache_dir)
            embeddings_path = cached_path(f"{base_path}{embeddings}/{model}_s{size}.zip", cache_dir=cache_dir)

        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings_path = Path(embeddings)

        self.name: str = str(embeddings_path)
        self.static_embeddings = True

        log.info("Reading embeddings from %s" % embeddings_path)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            open_inside_zip(str(embeddings_path), cache_dir=cache_dir)
        )

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super(TokenEmbeddings, self).__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name


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
