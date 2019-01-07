import re
import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Dict

import gensim
import numpy as np
import torch
from deprecated import deprecated

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PRETRAINED_MODEL_ARCHIVE_MAP

from .nn import LockedDropout, WordDropout
from .data import Dictionary, Token, Sentence
from .file_utils import cached_path


log = logging.getLogger('flair')


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == 'word-level':
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys(): everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return 'word-level'


class DocumentEmbeddings(Embeddings):
    """Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return 'sentence-level'


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings], detach: bool = True):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module('list_embedding_{}'.format(i), embedding)

        self.detach: bool = detach
        self.name: str = 'Stack'
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


class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        old_base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'
        base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/'
        embeddings_path_v4 = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/'

        cache_dir = Path('embeddings')

        # GLOVE embeddings
        if embeddings.lower() == 'glove' or embeddings.lower() == 'en-glove':
            cached_path(f'{old_base_path}glove.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}glove.gensim', cache_dir=cache_dir)

        # KOMNIOS embeddings
        elif embeddings.lower() == 'extvec' or embeddings.lower() == 'en-extvec':
            cached_path(f'{old_base_path}extvec.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}extvec.gensim', cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == 'crawl' or embeddings.lower() == 'en-crawl':
            cached_path(f'{base_path}en-fasttext-crawl-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}en-fasttext-crawl-300d-1M', cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == 'news' or embeddings.lower() == 'en-news' or embeddings.lower() == 'en':
            cached_path(f'{base_path}en-fasttext-news-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}en-fasttext-news-300d-1M', cache_dir=cache_dir)

        # twitter embeddings
        elif embeddings.lower() == 'twitter' or embeddings.lower() == 'en-twitter':
            cached_path(f'{old_base_path}twitter.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}twitter.gensim', cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(f'{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M', cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith('-wiki'):
            cached_path(f'{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M', cache_dir=cache_dir)

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith('-crawl'):
            cached_path(f'{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M',
                                     cache_dir=cache_dir)

        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(str(embeddings))

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif re.sub(r'\d', '#', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '#', word.lower())]
                elif re.sub(r'\d', '0', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '0', word.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


class ELMoEmbeddings(TokenEmbeddings):
    """Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018."""

    def __init__(self, model: str = 'original'):
        super().__init__()

        try:
            import allennlp.commands.elmo
        except:
            log.warning('-' * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning('To use ELMoEmbeddings, please first install with "pip install allennlp"')
            log.warning('-' * 100)
            pass

        self.name = 'elmo-' + model
        self.static_embeddings = True

        # the default model for ELMo is the 'original' model, which is very large
        options_file = allennlp.commands.elmo.DEFAULT_OPTIONS_FILE
        weight_file = allennlp.commands.elmo.DEFAULT_WEIGHT_FILE
        # alternatively, a small, medium or portuguese model can be selected by passing the appropriate mode name
        if model == 'small':
            options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
            weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        if model == 'medium':
            options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'
            weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
        if model == 'pt' or model == 'portuguese':
            options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json'
            weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5'

        # put on Cuda if available
        cuda_device = 0 if torch.cuda.is_available() else -1
        self.ee = allennlp.commands.elmo.ElmoEmbedder(options_file=options_file,
                                                      weight_file=weight_file,
                                                      cuda_device=cuda_device)

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])

        embeddings = self.ee.embed_batch(sentence_words)

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                embedding = torch.cat([torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                                       torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                                       torch.FloatTensor(sentence_embeddings[2, token_idx, :])], 0)

                word_embedding = torch.autograd.Variable(embedding)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name

class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(self, path_to_char_dict: str = None):
        """Uses the default character dictionary if none provided."""

        super(CharacterEmbeddings, self).__init__()
        self.name = 'Char'
        self.static_embeddings = False

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load('common-chars')
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(path_to_char_dict)

        self.char_embedding_dim: int = 25
        self.hidden_size_char: int = 25
        self.char_embedding = torch.nn.Embedding(len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(self.char_embedding_dim, self.hidden_size_char, num_layers=1,
                                      bidirectional=True)

        self.__embedding_length = self.char_embedding_dim * 2

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                token: Token = token
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
            tokens_mask = np.zeros((len(tokens_sorted_by_length), longest_token_in_sentence), dtype='int')
            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, :chars2_length[i]] = c

            tokens_mask = torch.LongTensor(tokens_mask)

            # chars for rnn processing
            chars = tokens_mask
            if torch.cuda.is_available():
                chars = chars.cuda()

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(character_embeddings, chars2_length)

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2))))
            if torch.cuda.is_available():
                chars_embeds_temp = chars_embeds_temp.cuda()
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

    def __init__(self, model: str, detach: bool = True, use_cache: bool = False, cache_directory: Path = None):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param detach: if set to False, the gradient will propagate into the language model. this dramatically slows down
                training and often leads to worse results, so not recommended.
        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will
                not allow re-use of once computed embeddings that do not fit into memory
        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache
                is written to the provided directory.
        """
        super().__init__()

        cache_dir = Path('embeddings')

        # multilingual forward (English, German, French, Italian, Dutch, Polish)
        if model.lower() == 'multi-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # multilingual backward  (English, German, French, Italian, Dutch, Polish)
        elif model.lower() == 'multi-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # multilingual forward fast (English, German, French, Italian, Dutch, Polish)
        elif model.lower() == 'multi-forward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # multilingual backward fast (English, German, French, Italian, Dutch, Polish)
        elif model.lower() == 'multi-backward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-forward
        elif model.lower() == 'news-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-backward
        elif model.lower() == 'news-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-forward
        elif model.lower() == 'news-forward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-backward
        elif model.lower() == 'news-backward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-english-forward
        elif model.lower() == 'mix-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-english-backward
        elif model.lower() == 'mix-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-german-forward
        elif model.lower() == 'german-forward' or model.lower() == 'de-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-german-backward
        elif model.lower() == 'german-backward' or model.lower() == 'de-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # common crawl Polish forward
        elif model.lower() == 'polish-forward' or model.lower() == 'pl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # common crawl Polish backward
        elif model.lower() == 'polish-backward' or model.lower() == 'pl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Slovenian forward
        elif model.lower() == 'slovenian-forward' or model.lower() == 'sl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Slovenian backward
        elif model.lower() == 'slovenian-backward' or model.lower() == 'sl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Bulgarian forward
        elif model.lower() == 'bulgarian-forward' or model.lower() == 'bg-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Bulgarian backward
        elif model.lower() == 'bulgarian-backward' or model.lower() == 'bg-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Dutch forward
        elif model.lower() == 'dutch-forward' or model.lower() == 'nl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Dutch backward
        elif model.lower() == 'dutch-backward' or model.lower() == 'nl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Swedish forward
        elif model.lower() == 'swedish-forward' or model.lower() == 'sv-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Swedish backward
        elif model.lower() == 'swedish-backward' or model.lower() == 'sv-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # French forward
        elif model.lower() == 'french-forward' or model.lower() == 'fr-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # French backward
        elif model.lower() == 'french-backward' or model.lower() == 'fr-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Czech forward
        elif model.lower() == 'czech-forward' or model.lower() == 'cs-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Czech backward
        elif model.lower() == 'czech-backward' or model.lower() == 'cs-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Portuguese forward
        elif model.lower() == 'portuguese-forward' or model.lower() == 'pt-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Portuguese backward
        elif model.lower() == 'portuguese-backward' or model.lower() == 'pt-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Basque forward
        elif model.lower() == 'basque-forward' or model.lower() == 'eu-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-eu-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Basque backward
        elif model.lower() == 'basque-backward' or model.lower() == 'eu-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-eu-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)


        elif not Path(model).exists():
            raise ValueError(f'The given model "{model}" is not available or is not a valid path.')

        self.name = str(model)
        self.static_embeddings = detach

        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach

        self.is_forward_lm: bool = self.lm.is_forward_lm

        # initialize cache if use_cache set
        self.cache = None
        if use_cache:
            cache_path = Path(f'{self.name}-tmp-cache.sqllite') if not cache_directory else \
                cache_directory / f'{self.name}-tmp-cache.sqllite'
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(str(cache_path), autocommit=True)

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

        # set to eval mode
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state['cache'] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # if cache is used, try setting embeddings from cache first
        if 'cache' in self.__dict__ and self.cache is not None:

            # try populating embeddings from cache
            all_embeddings_retrieved_from_cache: bool = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)

                if not embeddings:
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for token, embedding in zip(sentence, embeddings):
                        token.set_embedding(self.name, torch.FloatTensor(embedding))

            if all_embeddings_retrieved_from_cache:
                return sentences

        with torch.no_grad():

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences]

            longest_character_sequence_in_batch: int = len(max(text_sentences, key=len))

            # pad strings with whitespaces to longest sentence
            sentences_padded: List[str] = []
            append_padded_sentence = sentences_padded.append

            start_marker = '\n'

            end_marker = ' '
            extra_offset = len(start_marker)
            for sentence_text in text_sentences:
                pad_by = longest_character_sequence_in_batch - len(sentence_text)
                if self.is_forward_lm:
                    padded = '{}{}{}{}'.format(start_marker, sentence_text, end_marker, pad_by * ' ')
                    append_padded_sentence(padded)
                else:
                    padded = '{}{}{}{}'.format(start_marker, sentence_text[::-1], end_marker, pad_by * ' ')
                    append_padded_sentence(padded)

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()

                offset_forward: int = extra_offset
                offset_backward: int = len(sentence_text) + extra_offset

                for token in sentence.tokens:
                    token: Token = token

                    offset_forward += len(token.text)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    # if self.tokenized_lm or token.whitespace_after:
                    offset_forward += 1
                    offset_backward -= 1

                    offset_backward -= len(token.text)

                    token.set_embedding(self.name, embedding)

        if 'cache' in self.__dict__ and self.cache is not None:
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [token._embeddings[self.name].tolist() for token in
                                                              sentence]

        return sentences

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):

    def __init__(self,
                 contextual_embeddings: Union[str, FlairEmbeddings],
                 pooling: str = 'fade',
                 only_capitalized: bool = False,
                 **kwargs):

        super().__init__()

        # use the character language model embeddings as basis
        if type(contextual_embeddings) is str:
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(contextual_embeddings, **kwargs)
        else:
            self.context_embeddings: FlairEmbeddings = contextual_embeddings

        # length is twice the original character LM embedding length
        self.embedding_length = self.context_embeddings.embedding_length * 2
        self.name = self.context_embeddings.name + '-context'

        # these fields are for the embedding memory
        self.word_embeddings = {}
        self.word_count = {}

        # whether to add only capitalized words to memory (faster runtime and lower memory consumption)
        self.only_capitalized = only_capitalized

        # we re-compute embeddings dynamically at each epoch
        self.static_embeddings = False

        # set the memory method
        self.pooling = pooling
        if pooling == 'mean':
            self.aggregate_op = torch.add
        elif pooling == 'fade':
            self.aggregate_op = torch.add
        elif pooling == 'max':
            self.aggregate_op = torch.max
        elif pooling == 'min':
            self.aggregate_op = torch.min

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            # memory is wiped each time we do a training run
            print('train mode resetting embeddings')
            self.word_embeddings = {}
            self.word_count = {}

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.context_embeddings.embed(sentences)

        # if we keep a pooling, it needs to be updated continuously
        for sentence in sentences:
            for token in sentence.tokens:

                # update embedding
                local_embedding = token._embeddings[self.context_embeddings.name]

                if token.text[0].isupper() or not self.only_capitalized:

                    if token.text not in self.word_embeddings:
                        self.word_embeddings[token.text] = local_embedding
                        self.word_count[token.text] = 1
                    else:
                        aggregated_embedding = self.aggregate_op(self.word_embeddings[token.text], local_embedding)
                        if self.pooling == 'fade':
                            aggregated_embedding /= 2
                        self.word_embeddings[token.text] = aggregated_embedding
                        self.word_count[token.text] += 1

        # add embeddings after updating
        for sentence in sentences:
            for token in sentence.tokens:
                if token.text in self.word_embeddings:
                    base = self.word_embeddings[token.text] / self.word_count[token.text] \
                        if self.pooling == 'mean' else self.word_embeddings[token.text]
                else:
                    base = token._embeddings[self.context_embeddings.name]

                token.set_embedding(self.name, base)

        return sentences

    def embedding_length(self) -> int:
        return self.embedding_length


class BertEmbeddings(TokenEmbeddings):

    def __init__(self,
                 bert_model: str = 'bert-base-uncased',
                 layers: str = '-1,-2,-3,-4',
                 pooling_operation: str = 'first'):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model: name of BERT model ('')
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        if bert_model not in PRETRAINED_MODEL_ARCHIVE_MAP.keys():
            raise ValueError('Provided bert-model is not available.')

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, token_subtoken_count):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(self, sentences, max_sequence_length: int) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0:(max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(BertEmbeddings.BertInputFeatures(
                unique_id=sentence_index,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                token_subtoken_count=token_subtoken_count))

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max([self.tokenizer.tokenize(sentence.to_tokenized_string()) for sentence in sentences], key=len))

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(sentences, longest_sentence_in_batch)
        if torch.cuda.is_available():
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
            all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        else:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.eval()
        all_encoder_layers, _ = self.model(all_input_ids, token_type_ids=None, attention_mask=all_input_masks)

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu()[sentence_index]
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == 'first':
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[token_idx:token_idx + feature.token_subtoken_count[token.idx]]
                        embeddings = [embedding.unsqueeze(0) for embedding in embeddings]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return len(self.layer_indexes) * self.model.config.hidden_size


class CharLMEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018. """

    @deprecated(version='0.4', reason="Use 'FlairEmbeddings' instead.")
    def __init__(self, model: str, detach: bool = True, use_cache: bool = False, cache_directory: Path = None):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param detach: if set to False, the gradient will propagate into the language model. this dramatically slows down
                training and often leads to worse results, so not recommended.
        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will
                not allow re-use of once computed embeddings that do not fit into memory
        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache
                is written to the provided directory.
        """
        super().__init__()

        cache_dir = Path('embeddings')

        # multilingual forward (English, German, French, Italian, Dutch, Polish)
        if model.lower() == 'multi-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # multilingual backward  (English, German, French, Italian, Dutch, Polish)
        elif model.lower() == 'multi-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-forward
        elif model.lower() == 'news-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-backward
        elif model.lower() == 'news-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-forward
        elif model.lower() == 'news-forward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # news-english-backward
        elif model.lower() == 'news-backward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-english-forward
        elif model.lower() == 'mix-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-english-backward
        elif model.lower() == 'mix-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-german-forward
        elif model.lower() == 'german-forward' or model.lower() == 'de-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # mix-german-backward
        elif model.lower() == 'german-backward' or model.lower() == 'de-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # common crawl Polish forward
        elif model.lower() == 'polish-forward' or model.lower() == 'pl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # common crawl Polish backward
        elif model.lower() == 'polish-backward' or model.lower() == 'pl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Slovenian forward
        elif model.lower() == 'slovenian-forward' or model.lower() == 'sl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Slovenian backward
        elif model.lower() == 'slovenian-backward' or model.lower() == 'sl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Bulgarian forward
        elif model.lower() == 'bulgarian-forward' or model.lower() == 'bg-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Bulgarian backward
        elif model.lower() == 'bulgarian-backward' or model.lower() == 'bg-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Dutch forward
        elif model.lower() == 'dutch-forward' or model.lower() == 'nl-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Dutch backward
        elif model.lower() == 'dutch-backward' or model.lower() == 'nl-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Swedish forward
        elif model.lower() == 'swedish-forward' or model.lower() == 'sv-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Swedish backward
        elif model.lower() == 'swedish-backward' or model.lower() == 'sv-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # French forward
        elif model.lower() == 'french-forward' or model.lower() == 'fr-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # French backward
        elif model.lower() == 'french-backward' or model.lower() == 'fr-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Czech forward
        elif model.lower() == 'czech-forward' or model.lower() == 'cs-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Czech backward
        elif model.lower() == 'czech-backward' or model.lower() == 'cs-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        # Portuguese forward
        elif model.lower() == 'portuguese-forward' or model.lower() == 'pt-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        # Portuguese backward
        elif model.lower() == 'portuguese-backward' or model.lower() == 'pt-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)

        elif not Path(model).exists():
            raise ValueError(f'The given model "{model}" is not available or is not a valid path.')

        self.name = str(model)
        self.static_embeddings = detach

        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach

        self.is_forward_lm: bool = self.lm.is_forward_lm

        # initialize cache if use_cache set
        self.cache = None
        if use_cache:
            cache_path = Path(f'{self.name}-tmp-cache.sqllite') if not cache_directory else \
                cache_directory / f'{self.name}-tmp-cache.sqllite'
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(str(cache_path), autocommit=True)

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

        # set to eval mode
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state['cache'] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # if cache is used, try setting embeddings from cache first
        if 'cache' in self.__dict__ and self.cache is not None:

            # try populating embeddings from cache
            all_embeddings_retrieved_from_cache: bool = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)

                if not embeddings:
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for token, embedding in zip(sentence, embeddings):
                        token.set_embedding(self.name, torch.FloatTensor(embedding))

            if all_embeddings_retrieved_from_cache:
                return sentences

        # if this is not possible, use LM to generate embedding. First, get text sentences
        text_sentences = [sentence.to_tokenized_string() for sentence in sentences]

        longest_character_sequence_in_batch: int = len(max(text_sentences, key=len))

        # pad strings with whitespaces to longest sentence
        sentences_padded: List[str] = []
        append_padded_sentence = sentences_padded.append

        end_marker = ' '
        extra_offset = 1
        for sentence_text in text_sentences:
            pad_by = longest_character_sequence_in_batch - len(sentence_text)
            if self.is_forward_lm:
                padded = '\n{}{}{}'.format(sentence_text, end_marker, pad_by * ' ')
                append_padded_sentence(padded)
            else:
                padded = '\n{}{}{}'.format(sentence_text[::-1], end_marker, pad_by * ' ')
                append_padded_sentence(padded)

        # get hidden states from language model
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

        # take first or last hidden states from language model as word representation
        for i, sentence in enumerate(sentences):
            sentence_text = sentence.to_tokenized_string()

            offset_forward: int = extra_offset
            offset_backward: int = len(sentence_text) + extra_offset

            for token in sentence.tokens:
                token: Token = token

                offset_forward += len(token.text)

                if self.is_forward_lm:
                    offset = offset_forward
                else:
                    offset = offset_backward

                embedding = all_hidden_states_in_lm[offset, i, :]

                # if self.tokenized_lm or token.whitespace_after:
                offset_forward += 1
                offset_backward -= 1

                offset_backward -= len(token.text)

                token.set_embedding(self.name, embedding)

        if 'cache' in self.__dict__ and self.cache is not None:
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [token._embeddings[self.name].tolist() for token in
                                                              sentence]

        return sentences

    def __str__(self):
        return self.name


class DocumentMeanEmbeddings(DocumentEmbeddings):

    @deprecated(version='0.3.1', reason="The functionality of this class is moved to 'DocumentPoolEmbeddings'")
    def __init__(self, token_embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=token_embeddings)
        self.name: str = 'document_mean'

        self.__embedding_length: int = self.embeddings.embedding_length

        if torch.cuda.is_available():
            self.cuda()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        everything_embedded: bool = True

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:

            self.embeddings.embed(sentences)

            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    token: Token = token
                    word_embeddings.append(token.get_embedding().unsqueeze(0))

                word_embeddings = torch.cat(word_embeddings, dim=0)
                if torch.cuda.is_available():
                    word_embeddings = word_embeddings.cuda()

                mean_embedding = torch.mean(word_embeddings, 0)

                sentence.set_embedding(self.name, mean_embedding.unsqueeze(0))

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentPoolEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings: List[TokenEmbeddings], mode: str = 'mean'):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param mode: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.__embedding_length: int = self.embeddings.embedding_length

        if torch.cuda.is_available():
            self.cuda()

        self.mode = mode
        if self.mode == 'mean':
            self.pool_op = torch.mean
        elif mode == 'max':
            self.pool_op = torch.max
        elif mode == 'min':
            self.pool_op = torch.min
        else:
            raise ValueError(f'Pooling operation for {self.mode!r} is not defined')
        self.name: str = f'document_{self.mode}'

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        everything_embedded: bool = True

        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:

            self.embeddings.embed(sentences)

            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    token: Token = token
                    word_embeddings.append(token.get_embedding().unsqueeze(0))

                word_embeddings = torch.cat(word_embeddings, dim=0)
                if torch.cuda.is_available():
                    word_embeddings = word_embeddings.cuda()

                if self.mode == 'mean':
                    pooled_embedding = self.pool_op(word_embeddings, 0)
                else:
                    pooled_embedding, _ = self.pool_op(word_embeddings, 0)

                sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentLSTMEmbeddings(DocumentEmbeddings):

    def __init__(self,
                 embeddings: List[TokenEmbeddings],
                 hidden_size=128,
                 rnn_layers=1,
                 reproject_words: bool = True,
                 reproject_words_dimension: int = None,
                 bidirectional: bool = False,
                 dropout: float = 0.5,
                 word_dropout: float = 0.0,
                 locked_dropout: float = 0.0):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the lstm
        :param rnn_layers: the number of layers for the lstm
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the lstm or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.name = 'document_lstm'
        self.static_embeddings = False

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        # bidirectional LSTM on top of embedding layer
        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings,
                                                     self.embeddings_dimension)
        self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_size, num_layers=rnn_layers,
                                bidirectional=self.bidirectional)

        # dropouts
        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        if torch.cuda.is_available():
            self.cuda()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        # first, sort sentences by number of tokens
        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.FloatTensor(np.zeros(self.length_of_all_token_embeddings, dtype='float')).unsqueeze(0))

            word_embeddings_tensor = torch.cat(word_embeddings, 0)

            sentence_states = word_embeddings_tensor

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        sentence_tensor = torch.cat(all_sentence_tensors, 1)
        if torch.cuda.is_available():
            sentence_tensor = sentence_tensor.cuda()

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        sentence_tensor = self.dropout(sentence_tensor)

        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

        self.rnn.flatten_parameters()

        lstm_out, hidden = self.rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

        outputs = self.dropout(outputs)

        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM LSTM
        # --------------------------------------------------------------------
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[length - 1, sentence_no]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[0, sentence_no]
                embedding = torch.cat([first_rep, last_rep], 0)

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentLMEmbeddings(DocumentEmbeddings):
    def __init__(self, flair_embeddings: List[FlairEmbeddings], detach: bool = True):
        super().__init__()

        self.embeddings = flair_embeddings
        self.name = 'document_lm'

        self.static_embeddings = detach
        self.detach = detach

        self._embedding_length: int = sum(embedding.embedding_length for embedding in flair_embeddings)

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

                # if its a forward LM, take last state
                if embedding.is_forward_lm:
                    sentence.set_embedding(embedding.name, sentence[len(sentence)]._embeddings[embedding.name])
                else:
                    sentence.set_embedding(embedding.name, sentence[1]._embeddings[embedding.name])

        return sentences
