import hashlib
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Dict
from collections import Counter
from functools import lru_cache

import torch
from bpemb import BPEmb
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer, AutoTokenizer, AutoConfig, AutoModel

import flair
import gensim
import os
import re
import logging
import numpy as np

from flair.data import Sentence, Token, Corpus, Dictionary
from flair.embeddings.base import Embeddings, ScalarMix
from flair.file_utils import cached_path, open_inside_zip

log = logging.getLogger("flair")


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


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

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
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

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        self.embeddings = embeddings

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/token"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{hu_path}/glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/glove.gensim", cache_dir=cache_dir)

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(f"{hu_path}/turian.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/turian", cache_dir=cache_dir)

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(f"{hu_path}/extvec.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/extvec.gensim", cache_dir=cache_dir)

        # pubmed embeddings
        elif embeddings.lower() == "pubmed" or embeddings.lower() == "en-pubmed":
            cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() in["news", "en-news", "en"]:
            cached_path(f"{hu_path}/en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/en-fasttext-news-300d-1M", cache_dir=cache_dir)

        # twitter embeddings
        elif embeddings.lower() in ["twitter", "en-twitter"]:
            cached_path(f"{hu_path}/twitter.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/twitter.gensim", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M",  cache_dir=cache_dir)

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M", cache_dir=cache_dir)

        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )

        self.name: str = str(embeddings)
        self.static_embeddings = True

        if str(embeddings).endswith(".bin"):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True
            )
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings)
            )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) -> torch.Tensor:
        if word in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word]
        elif word.lower() in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word.lower()]
        elif re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "#", word.lower())
            ]
        elif re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "0", word.lower())
            ]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype="float")

        word_embedding = torch.tensor(
            word_embedding.tolist(), device=flair.device, dtype=torch.float
        )
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(word=word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        # fix serialized models
        if "embeddings" not in self.__dict__:
            self.embeddings = self.name

        return f"'{self.embeddings}'"

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

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(
                path_to_char_dict
            )

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )
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
                char_indices = [
                    self.char_dictionary.get_idx_for_item(char) for char in token.text
                ]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=lambda p: len(p), reverse=True
            )
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
                tokens_mask[i, : chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device
                )

            # chars for rnn processing
            chars = tokens_mask

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length
            )

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

    def __init__(self,
                 model,
                 fine_tune: bool = False,
                 chars_per_chunk: int = 512,
                 with_whitespace: bool = True,
                 tokenized_lm: bool = True,
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

        cache_dir = Path("embeddings")

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/flair"
        clef_hipe_path: str = "https://files.ifi.uzh.ch/cl/siclemat/impresso/clef-hipe-2020/flair"

        self.PRETRAINED_MODEL_ARCHIVE_MAP = {
            # multilingual models
            "multi-forward": f"{hu_path}/lm-jw300-forward-v0.1.pt",
            "multi-backward": f"{hu_path}/lm-jw300-backward-v0.1.pt",
            "multi-v0-forward": f"{hu_path}/lm-multi-forward-v0.1.pt",
            "multi-v0-backward": f"{hu_path}/lm-multi-backward-v0.1.pt",
            "multi-v0-forward-fast": f"{hu_path}/lm-multi-forward-fast-v0.1.pt",
            "multi-v0-backward-fast": f"{hu_path}/lm-multi-backward-fast-v0.1.pt",
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
            "ml-forward": f"https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-forward.pt",
            "ml-backward": f"https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-backward.pt",
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
            # CLEF HIPE Shared task
            "de-impresso-hipe-v1-forward": f"{clef_hipe_path}/de-hipe-flair-v1-forward/best-lm.pt",
            "de-impresso-hipe-v1-backward": f"{clef_hipe_path}/de-hipe-flair-v1-backward/best-lm.pt",
            "en-impresso-hipe-v1-forward": f"{clef_hipe_path}/en-flair-v1-forward/best-lm.pt",
            "en-impresso-hipe-v1-backward": f"{clef_hipe_path}/en-flair-v1-backward/best-lm.pt",
            "fr-impresso-hipe-v1-forward": f"{clef_hipe_path}/fr-hipe-flair-v1-forward/best-lm.pt",
            "fr-impresso-hipe-v1-backward": f"{clef_hipe_path}/fr-hipe-flair-v1-backward/best-lm.pt",
        }

        if type(model) == str:

            # load model if in pretrained model map
            if model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]

                # Fix for CLEF HIPE models (avoid overwriting best-lm.pt in cache_dir)
                if "impresso-hipe" in model.lower():
                    cache_dir = cache_dir / model.lower()
                model = cached_path(base_path, cache_dir=cache_dir)

            elif replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[
                    replace_with_language_code(model)
                ]
                model = cached_path(base_path, cache_dir=cache_dir)

            elif not Path(model).exists():
                raise ValueError(
                    f'The given model "{model}" is not available or is not a valid path.'
                )

        from flair.models import LanguageModel

        if type(model) == LanguageModel:
            self.lm: LanguageModel = model
            self.name = f"Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}"
        else:
            self.lm: LanguageModel = LanguageModel.load_language_model(model)
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
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

        # set to eval mode
        self.eval()

    def train(self, mode=True):

        # make compatible with serialized models (TODO: remove)
        if "fine_tune" not in self.__dict__:
            self.fine_tune = False
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

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

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences] if self.tokenized_lm \
                else [sentence.to_plain_string() for sentence in sentences]

            start_marker = self.lm.document_delimiter if "document_delimiter" in self.lm.__dict__ else '\n'
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

                    # only clone if optimization mode is 'gpu'
                    if flair.embedding_storage_mode == "gpu":
                        embedding = embedding.clone()

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

        # use the character language model embeddings as basis
        if type(contextual_embeddings) is str:
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(
                contextual_embeddings, **kwargs
            )
        else:
            self.context_embeddings: FlairEmbeddings = contextual_embeddings

        # length is twice the original character LM embedding length
        self.embedding_length = self.context_embeddings.embedding_length * 2
        self.name = self.context_embeddings.name + "-context"

        # these fields are for the embedding memory
        self.word_embeddings = {}
        self.word_count = {}

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
        return self.embedding_length

    def get_names(self) -> List[str]:
        return [self.name, self.context_embeddings.name]

    def __setstate__(self, d):
        self.__dict__ = d

        if flair.device != 'cpu':
            for key in self.word_embeddings:
                self.word_embeddings[key] = self.word_embeddings[key].cpu()


class TransformerWordEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        model: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        batch_size: int = 1,
        use_scalar_mix: bool = False,
        fine_tune: bool = False,
        allow_long_sentences: bool = True,
        **kwargs
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        """
        super().__init__()

        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
        self.model = AutoModel.from_pretrained(model, config=config, **kwargs)

        self.allow_long_sentences = allow_long_sentences

        if allow_long_sentences:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = self.tokenizer.model_max_length//2
        else:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = 0

        # model name
        self.name = 'transformer-word-' + str(model)

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
        # self.mix = ScalarMix(mixture_size=len(self.layer_indexes), trainable=False)
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size

        self.special_tokens = []
        # check if special tokens exist to circumvent error message
        if self.tokenizer._bos_token:
            self.special_tokens.append(self.tokenizer.bos_token)
        if self.tokenizer._cls_token:
            self.special_tokens.append(self.tokenizer.cls_token)

        # most models have an intial BOS token, except for XLNet, T5 and GPT2
        self.begin_offset = 1
        if type(self.tokenizer) == XLNetTokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == T5Tokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == GPT2Tokenizer:
            self.begin_offset = 0

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences."""

        # split into micro batches of size self.batch_size before pushing through transformer
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        # embed each micro-batch
        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    @staticmethod
    def _remove_special_markup(text: str):
        # remove special markup
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models
        return text

    def _get_processed_token_text(self, token: Token) -> str:
        pieces = self.tokenizer.tokenize(token.text)
        token_text = ''
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""

        # first, subtokenize each sentence and find out into how many subtokens each token was divided
        subtokenized_sentences = []
        subtokenized_sentences_token_lengths = []

        sentence_parts_lengths = []

        # TODO: keep for backwards compatibility, but remove in future
        # some pretrained models do not have this property, applying default settings now.
        # can be set manually after loading the model.
        if not hasattr(self, 'max_subtokens_sequence_length'):
            self.max_subtokens_sequence_length = None
            self.allow_long_sentences = False
            self.stride = 0

        non_empty_sentences = []
        empty_sentences = []

        for sentence in sentences:
            tokenized_string = sentence.to_tokenized_string()

            # method 1: subtokenize sentence
            # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

            # method 2:
            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)
            if len(subtokenized_sentence) == 0:
                empty_sentences.append(sentence)
                continue
            else:
                non_empty_sentences.append(sentence)

            token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sentence, subtokenized_sentence)
            subtokenized_sentences_token_lengths.append(token_subtoken_lengths)

            subtoken_ids_sentence = self.tokenizer.convert_tokens_to_ids(subtokenized_sentence)

            nr_sentence_parts = 0

            while subtoken_ids_sentence:
                nr_sentence_parts += 1
                encoded_inputs = self.tokenizer.encode_plus(subtoken_ids_sentence,
                                                            max_length=self.max_subtokens_sequence_length,
                                                            stride=self.stride,
                                                            return_overflowing_tokens=self.allow_long_sentences,
                                                            truncation=True,
                                                            )

                subtoken_ids_split_sentence = encoded_inputs['input_ids']
                subtokenized_sentences.append(torch.tensor(subtoken_ids_split_sentence, dtype=torch.long))

                if 'overflowing_tokens' in encoded_inputs:
                    subtoken_ids_sentence = encoded_inputs['overflowing_tokens']
                else:
                    subtoken_ids_sentence = None

            sentence_parts_lengths.append(nr_sentence_parts)

        # empty sentences get zero embeddings
        for sentence in empty_sentences:
            for token in sentence:
                token.set_embedding(self.name, torch.zeros(self.embedding_length))

        # only embed non-empty sentences and if there is at least one
        sentences = non_empty_sentences
        if len(sentences) == 0: return

        # find longest sentence in batch
        longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

        total_sentence_parts = sum(sentence_parts_lengths)
        # initialize batch tensors and mask
        input_ids = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=flair.device,
        )
        mask = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=flair.device,
        )
        for s_id, sentence in enumerate(subtokenized_sentences):
            sequence_length = len(sentence)
            input_ids[s_id][:sequence_length] = sentence
            mask[s_id][:sequence_length] = torch.ones(sequence_length)

        # put encoded batch through transformer model to get all hidden states of all encoder layers
        hidden_states = self.model(input_ids, attention_mask=mask)[-1]
        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)

        sentence_idx_offset = 0

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # iterate over all subtokenized sentences
            for sentence_idx, (sentence, subtoken_lengths, nr_sentence_parts) in enumerate(zip(sentences, subtokenized_sentences_token_lengths, sentence_parts_lengths)):

                sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]

                for i in range(1, nr_sentence_parts):
                    sentence_idx_offset += 1
                    remainder_sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]
                    # remove stride_size//2 at end of sentence_hidden_state, and half at beginning of remainder,
                    # in order to get some context into the embeddings of these words.
                    # also don't include the embedding of the extra [CLS] and [SEP] tokens.
                    sentence_hidden_state = torch.cat((sentence_hidden_state[:, :-1-self.stride//2, :],
                                                       remainder_sentence_hidden_state[:, 1 + self.stride//2:, :]), 1)

                subword_start_idx = self.begin_offset

                # for each token, get embedding
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):

                    # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
                    if number_of_subtokens == 0:
                        token.set_embedding(self.name, torch.zeros(self.embedding_length))
                        continue

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    subtoken_embeddings: List[torch.FloatTensor] = []

                    # get states from all selected layers, aggregate with pooling operation
                    for layer in self.layer_indexes:
                        current_embeddings = sentence_hidden_state[layer][subword_start_idx:subword_end_idx]

                        if self.pooling_operation == "first":
                            final_embedding: torch.FloatTensor = current_embeddings[0]

                        if self.pooling_operation == "last":
                            final_embedding: torch.FloatTensor = current_embeddings[-1]

                        if self.pooling_operation == "first_last":
                            final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                        if self.pooling_operation == "mean":
                            all_embeddings: List[torch.FloatTensor] = [
                                embedding.unsqueeze(0) for embedding in current_embeddings
                            ]
                            final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                        subtoken_embeddings.append(final_embedding)

                    # use scalar mix of embeddings if so selected
                    if self.use_scalar_mix:
                        sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                        # sm_embeddings = self.mix(subtoken_embeddings)

                        subtoken_embeddings = [sm_embeddings]

                    # set the extracted embedding for the token
                    token.set_embedding(self.name, torch.cat(subtoken_embeddings))

                    subword_start_idx += number_of_subtokens

    def reconstruct_tokens_from_subtokens(self, sentence, subtokens):
        word_iterator = iter(sentence)
        token = next(word_iterator)
        token_text = self._get_processed_token_text(token)
        token_subtoken_lengths = []
        reconstructed_token = ''
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
                    if token_text.startswith(subtoken.lower()): break

            subtoken_count += 1

            # append subtoken to reconstruct token
            reconstructed_token = reconstructed_token + subtoken

            # check if reconstructed token is the same as current token
            if reconstructed_token.lower() == token_text:

                # if so, add subtoken count
                token_subtoken_lengths.append(subtoken_count)

                # reset subtoken count and reconstructed token
                reconstructed_token = ''
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
            if len(token_subtoken_lengths) == len(sentence): break
            token = next(word_iterator)

        # check if all tokens were matched to subtokens
        if token != sentence[-1]:
            log.error(f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'")
            log.error(f"Last matched: '{token}'")
            log.error(f"Last sentence: '{sentence[-1]}'")
            log.error(f"subtokenized: '{subtokens}'")
        return token_subtoken_lengths

    def train(self, mode=True):
        # if fine-tuning is not enabled (i.e. a "feature-based approach" used), this
        # module should never be in training mode
        if not self.fine_tune:
            pass
        else:
            super().train(mode)

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        if not self.use_scalar_mix:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size

        if self.pooling_operation == 'first_last': length *= 2

        return length

    def __setstate__(self, d):
        self.__dict__ = d

        # reload tokenizer to get around serialization issues
        model_name = self.name.split('transformer-word-')[-1]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            pass


class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings with oov functionality"""

    def __init__(self, embeddings: str, use_local: bool = True, field: str = None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
        """

        cache_dir = Path("embeddings")

        if use_local:
            if not Path(embeddings).exists():
                raise ValueError(
                    f'The given embeddings "{embeddings}" is not available or is not a valid path.'
                )
        else:
            embeddings = cached_path(f"{embeddings}", cache_dir=cache_dir)

        self.embeddings = embeddings

        self.name: str = str(embeddings)

        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(
            str(embeddings)
        )

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) -> torch.Tensor:
        try:
            word_embedding = self.precomputed_word_embeddings[word]
        except:
            word_embedding = np.zeros(self.embedding_length, dtype="float")

        word_embedding = torch.tensor(
            word_embedding.tolist(), device=flair.device, dtype=torch.float
        )
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
    """One-hot encoded embeddings. """

    def __init__(
        self,
        corpus: Corpus,
        field: str = "text",
        embedding_length: int = 300,
        min_freq: int = 3,
    ):
        """
        Initializes one-hot encoded word embeddings and a trainable embedding layer
        :param corpus: you need to pass a Corpus in order to construct the vocabulary
        :param field: by default, the 'text' of tokens is embedded, but you can also embed tags such as 'pos'
        :param embedding_length: dimensionality of the trainable embedding layer
        :param min_freq: minimum frequency of a word to become part of the vocabulary
        """
        super().__init__()
        self.name = "one-hot"
        self.static_embeddings = False
        self.min_freq = min_freq
        self.field = field

        tokens = list(map((lambda s: s.tokens), corpus.train))
        tokens = [token for sublist in tokens for token in sublist]

        if field == "text":
            most_common = Counter(list(map((lambda t: t.text), tokens))).most_common()
        else:
            most_common = Counter(
                list(map((lambda t: t.get_tag(field).value), tokens))
            ).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        self.vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)

        # max_tokens = 500
        self.__embedding_length = embedding_length

        print(self.vocab_dictionary.idx2item)
        print(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = torch.nn.Embedding(
            len(self.vocab_dictionary), self.__embedding_length
        )
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        one_hot_sentences = []
        for i, sentence in enumerate(sentences):

            if self.field == "text":
                context_idxs = [
                    self.vocab_dictionary.get_idx_for_item(t.text)
                    for t in sentence.tokens
                ]
            else:
                context_idxs = [
                    self.vocab_dictionary.get_idx_for_item(t.get_tag(self.field).value)
                    for t in sentence.tokens
                ]

            one_hot_sentences.extend(context_idxs)

        one_hot_sentences = torch.tensor(one_hot_sentences, dtype=torch.long).to(
            flair.device
        )

        embedded = self.embedding_layer.forward(one_hot_sentences)

        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "min_freq={}".format(self.min_freq)


class HashEmbeddings(TokenEmbeddings):
    """Standard embeddings with Hashing Trick."""

    def __init__(
        self, num_embeddings: int = 1000, embedding_length: int = 300, hash_method="md5"
    ):

        super().__init__()
        self.name = "hash"
        self.static_embeddings = False

        self.__num_embeddings = num_embeddings
        self.__embedding_length = embedding_length

        self.__hash_method = hash_method

        # model architecture
        self.embedding_layer = torch.nn.Embedding(
            self.__num_embeddings, self.__embedding_length
        )
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

        self.to(flair.device)

    @property
    def num_embeddings(self) -> int:
        return self.__num_embeddings

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        def get_idx_for_item(text):
            hash_function = hashlib.new(self.__hash_method)
            hash_function.update(bytes(str(text), "utf-8"))
            return int(hash_function.hexdigest(), 16) % self.__num_embeddings

        hash_sentences = []
        for i, sentence in enumerate(sentences):
            context_idxs = [get_idx_for_item(t.text) for t in sentence.tokens]

            hash_sentences.extend(context_idxs)

        hash_sentences = torch.tensor(hash_sentences, dtype=torch.long).to(flair.device)

        embedded = self.embedding_layer.forward(hash_sentences)

        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1

        return sentences

    def __str__(self):
        return self.name


class MuseCrosslingualEmbeddings(TokenEmbeddings):
    def __init__(self,):
        self.name: str = f"muse-crosslingual"
        self.static_embeddings = True
        self.__embedding_length: int = 300
        self.language_embeddings = {}
        super().__init__()

    @lru_cache(maxsize=10000, typed=False)
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
        word_embedding = torch.tensor(
            word_embedding, device=flair.device, dtype=torch.float
        )
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
                "pl",
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
                embeddings_file = cached_path(
                    f"{hu_path}/muse.{language_code}.vec.gensim", cache_dir=cache_dir
                )

                # load the model
                self.language_embeddings[
                    language_code
                ] = gensim.models.KeyedVectors.load(str(embeddings_file))

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(
                    language_code=language_code, word=word
                )

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
        self.cache_dir: Path = Path(flair.cache_root) / "embeddings"
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
        cache_dir= None,
        model_file_path: Path = None,
        embedding_file_path: Path = None,
        **kwargs,
    ):
        """
        Initializes BP embeddings. Constructor downloads required files if not there.
        """
        if not cache_dir:
            cache_dir = Path(flair.cache_root) / "embeddings"
        if language:
            self.name: str = f"bpe-{language}-{syllables}-{dim}"
        else:
            assert (
                model_file_path is not None and embedding_file_path is not None
            ), "Need to specify model_file_path and embedding_file_path if no language is given in BytePairEmbeddings(...)"
            dim=None

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
            self.name: str = f"bpe-custom-{self.embedder.vs}-{self.embedder.dim}"
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
                    token.set_embedding(
                        self.name, torch.zeros(self.embedding_length, dtype=torch.float)
                    )
                else:
                    # all other words get embedded
                    embeddings = self.embedder.embed(word.lower())
                    embedding = np.concatenate(
                        (embeddings[0], embeddings[len(embeddings) - 1])
                    )
                    token.set_embedding(
                        self.name, torch.tensor(embedding, dtype=torch.float)
                    )

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
        self, model: str = "original", options_file: str = None, weight_file: str = None, embedding_mode: str = "all"
    ):
        super().__init__()

        try:
            import allennlp.commands.elmo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                'To use ELMoEmbeddings, please first install with "pip install allennlp==0.9.0"'
            )
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
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5"
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
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length
    
    def use_layers_all(self,x):
        return torch.cat(x, 0)
    
    def use_layers_top(self,x):
        return x[-1]
    
    def use_layers_average(self,x):
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
                    torch.FloatTensor(sentence_embeddings[2, token_idx, :])
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
            [state.to(flair.device) for state in self.ee.elmo_bilm._elmo_lstm._states])


class NILCEmbeddings(WordEmbeddings):
    def __init__(self, embeddings: str, model: str = "skip", size: int = 100):
        """
        Initializes portuguese classic word embeddings trained by NILC Lab (http://www.nilc.icmc.usp.br/embeddings).
        Constructor downloads required files if not there.
        :param embeddings: one of: 'fasttext', 'glove', 'wang2vec' or 'word2vec'
        :param model: one of: 'skip' or 'cbow'. This is not applicable to glove.
        :param size: one of: 50, 100, 300, 600 or 1000.
        """

        base_path = "http://143.107.183.175:22980/download.php?file=embeddings/"

        cache_dir = Path("embeddings") / embeddings.lower()

        # GLOVE embeddings
        if embeddings.lower() == "glove":
            cached_path(
                f"{base_path}{embeddings}/{embeddings}_s{size}.zip", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}{embeddings}/{embeddings}_s{size}.zip", cache_dir=cache_dir
            )

        elif embeddings.lower() in ["fasttext", "wang2vec", "word2vec"]:
            cached_path(
                f"{base_path}{embeddings}/{model}_s{size}.zip", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}{embeddings}/{model}_s{size}.zip", cache_dir=cache_dir
            )

        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )

        self.name: str = str(embeddings)
        self.static_embeddings = True

        log.info("Reading embeddings from %s" % embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            open_inside_zip(str(embeddings), cache_dir=cache_dir)
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
