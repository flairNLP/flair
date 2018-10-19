import os
import re
from abc import abstractmethod
from typing import List, Union, Dict, Tuple

import gensim
import numpy as np
import torch

from .nn import LockedDropout, WordDropout
from .data import Dictionary, Token, Sentence
from .file_utils import cached_path


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


class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings):
        """Init one of: 'glove', 'extvec', 'ft-crawl', 'ft-german'.
        Constructor downloads required files if not there."""

        base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'

        # GLOVE embeddings
        if embeddings.lower() == 'glove' or embeddings.lower() == 'en-glove':
            cached_path(os.path.join(base_path, 'glove.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'glove.gensim'), cache_dir='embeddings')

        # twitter embeddings
        if embeddings.lower() == 'twitter' or embeddings.lower() == 'en-twitter':
            cached_path(os.path.join(base_path, 'twitter.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'twitter.gensim'), cache_dir='embeddings')

        # KOMNIOS embeddings
        if embeddings.lower() == 'extvec' or embeddings.lower() == 'en-extvec':
            cached_path(os.path.join(base_path, 'extvec.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'extvec.gensim'), cache_dir='embeddings')

        # NUMBERBATCH embeddings
        if embeddings.lower() == 'numberbatch' or embeddings.lower() == 'en-numberbatch':
            cached_path(os.path.join(base_path, 'numberbatch-en.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'numberbatch-en'), cache_dir='embeddings')

        # FT-CRAWL embeddings
        if embeddings.lower() == 'crawl' or embeddings.lower() == 'en-crawl':
            cached_path(os.path.join(base_path, 'ft-crawl.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'ft-crawl.gensim'), cache_dir='embeddings')

        # FT-CRAWL embeddings
        if embeddings.lower() == 'news' or embeddings.lower() == 'en-news':
            cached_path(os.path.join(base_path, 'ft-news.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'ft-news.gensim'), cache_dir='embeddings')

        # GERMAN FASTTEXT embeddings
        if embeddings.lower() == 'de-fasttext':
            cached_path(os.path.join(base_path, 'ft-wiki-de.gensim.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'ft-wiki-de.gensim'), cache_dir='embeddings')

        # NUMBERBATCH embeddings
        if embeddings.lower() == 'de-numberbatch':
            cached_path(os.path.join(base_path, 'de-numberbatch.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'de-numberbatch'), cache_dir='embeddings')

        # SWEDISCH FASTTEXT embeddings
        if embeddings.lower() == 'sv-fasttext':
            cached_path(os.path.join(base_path, 'cc.sv.300.vectors.npy'), cache_dir='embeddings')
            embeddings = cached_path(os.path.join(base_path, 'cc.sv.300'), cache_dir='embeddings')

        self.name = embeddings
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(embeddings)

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '#', token.text.lower())]
                elif re.sub('\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences


class MemoryEmbeddings(TokenEmbeddings):

    def __init__(self, tag_type: str, tag_dictionary: Dictionary):

        self.name = "memory"
        self.static_embeddings = False
        self.tag_type: str = tag_type
        self.tag_dictionary: Dictionary = tag_dictionary
        self.__embedding_length: int = len(tag_dictionary)

        self.memory: Dict[str:List] = {}

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            self.memory: Dict[str:List] = {}

    def update_embedding(self, text: str, tag: str):
        self.memory[text][self.tag_dictionary.get_idx_for_item(tag)] += 1

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if token.text not in self.memory:
                    self.memory[token.text] = [0] * self.__embedding_length

                word_embedding = torch.FloatTensor(self.memory[token.text])
                import torch.nn.functional as F
                word_embedding = F.normalize(word_embedding, p=2, dim=0)

                token.set_embedding(self.name, word_embedding)

                # add label if in training mode
                if self.training:
                    self.update_embedding(token.text, token.get_tag(self.tag_type).value)

        return sentences


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


class CharLMEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self, model, detach: bool = True, use_cache: bool = True, cache_directory: str = None):
        """
            Contextual string embeddings of words, as proposed in Akbik et al., 2018.

            Parameters
            ----------
            arg1 : model
                model string, one of 'news-forward', 'news-backward', 'mix-forward', 'mix-backward', 'german-forward',
                'german-backward' depending on which character language model is desired
            arg2 : detach
                if set to false, the gradient will propagate into the language model. this dramatically slows down
                training and often leads to worse results, so not recommended.
            arg3 : use_cache
                if set to false, will not write embeddings to file for later retrieval. this saves disk space but will
                not allow re-use of once computed embeddings that do not fit into memory
            arg3 : cache_directory
                if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache
                is written to the provided directory.
        """
        super().__init__()

        # news-english-forward
        if model.lower() == 'news-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # news-english-backward
        if model.lower() == 'news-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # news-english-forward
        if model.lower() == 'news-forward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # news-english-backward
        if model.lower() == 'news-backward-fast':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # mix-english-forward
        if model.lower() == 'mix-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # mix-english-backward
        if model.lower() == 'mix-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # mix-german-forward
        if model.lower() == 'german-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # mix-german-backward
        if model.lower() == 'german-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # common crawl Polish forward
        if model.lower() == 'polish-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        # common crawl Polish backward
        if model.lower() == 'polish-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir='embeddings')

        self.name = model
        self.static_embeddings = detach

        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach

        self.is_forward_lm: bool = self.lm.is_forward_lm

        # caching variables
        self.use_cache: bool = use_cache
        self.cache = None
        self.cache_directory: str = cache_directory

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state['cache'] = None
        state['use_cache'] = False
        state['cache_directory'] = None
        return state

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # this whole block is for compatibility with older serialized models  TODO: remove in version 0.4
        if 'cache' not in self.__dict__ or 'cache_directory' not in self.__dict__:
            self.use_cache = False
            self.cache_directory = None
        else:
            cache_path = '{}-tmp-cache.sqllite'.format(self.name) if not self.cache_directory else os.path.join(
                self.cache_directory, '{}-tmp-cache.sqllite'.format(os.path.basename(self.name)))
            if not os.path.exists(cache_path):
                self.use_cache = False
                self.cache_directory = None

        # if cache is used, try setting embeddings from cache first
        if self.use_cache:

            # lazy initialization of cache
            if not self.cache:
                from sqlitedict import SqliteDict
                self.cache = SqliteDict(cache_path, autocommit=True)

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

        if self.use_cache:
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [token._embeddings[self.name].tolist() for token in sentence]

        return sentences


class DocumentMeanEmbeddings(DocumentEmbeddings):

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


class DocumentLSTMEmbeddings(DocumentEmbeddings):

    def __init__(self,
                 token_embeddings: List[TokenEmbeddings],
                 hidden_states=128,
                 num_layers=1,
                 reproject_words: bool = True,
                 reproject_words_dimension: int = None,
                 bidirectional: bool = False,
                 use_word_dropout: bool = False,
                 use_locked_dropout: bool = False):
        """The constructor takes a list of embeddings to be combined.
        :param token_embeddings: a list of token embeddings
        :param hidden_states: the number of hidden states in the lstm
        :param num_layers: the number of layers for the lstm
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the lstm or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not
        representation of the lstm to be used as final document embedding.
        :param use_word_dropout: boolean value, indicating whether to use word dropout or not.
        :param use_locked_dropout: boolean value, indicating whether to use locked dropout or not.
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=token_embeddings)

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.name = 'document_lstm'
        self.static_embeddings = False

        self.__embedding_length: int = hidden_states
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        # bidirectional LSTM on top of embedding layer
        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings,
                                                     self.embeddings_dimension)
        self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_states, num_layers=num_layers,
                                 bidirectional=self.bidirectional)

        # dropouts
        if use_locked_dropout:
            self.dropout: torch.nn.Module = LockedDropout(0.5)
        else:
            self.dropout = torch.nn.Dropout(0.5)

        self.use_word_dropout: bool = use_word_dropout
        if self.use_word_dropout:
            self.word_dropout = WordDropout(0.05)

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
            last_rep = outputs[length - 1, sentence_no].unsqueeze(0)

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[0, sentence_no].unsqueeze(0)
                embedding = torch.cat([first_rep, last_rep], 1)

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class DocumentLMEmbeddings(DocumentEmbeddings):
    def __init__(self, charlm_embeddings: List[CharLMEmbeddings], detach: bool = True):
        super().__init__()

        self.embeddings = charlm_embeddings

        self.static_embeddings = detach
        self.detach = detach

        dummy: Sentence = Sentence('jo')
        self.embed([dummy])
        self._embedding_length: int = len(dummy.embedding)

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
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

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass
