import pickle
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import gensim
import numpy as np
import torch

from .file_utils import cached_path
from .language_model import RNNModel
from .data import Dictionary, Token, Sentence, TaggedCorpus


class TextEmbeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Ever new type of embedding must implement these methods."""

    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    def embed(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        everything_embedded: bool = True
        for sentence in sentences:
            for token in sentence.tokens:
                if self.name not in token._embeddings.keys(): everything_embedded = False

        # print(everything_embedded)
        if not everything_embedded or not self.static_embeddings:
            # print('retrieving embeddings %s' + self.name)
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class StackedEmbeddings(TextEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TextEmbeddings], detach: bool = True):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module('list_embedding_%s' % str(i), embedding)

        self.detach = detach
        self.name = 'Stack'
        self.static_embeddings = True

        self.embedding_length: int = 0
        for embedding in embeddings:
            self.embedding_length += embedding.embedding_length

    def embed(self, sentences: List[Sentence], static_embeddings: bool = True):

        for embedding in self.embeddings:
            embedding.embed(sentences)

    def embedding_length(self) -> int:
        return self.embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences


class WordEmbeddings(TextEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings):
        """Init one of: 'glove', 'extvec', 'ft-crawl', 'ft-german'.
        Constructor downloads required files if not there."""

        base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'

        # GLOVE embeddings
        if embeddings.lower() == 'glove':
            cached_path('%sglove.gensim.vectors.npy' % base_path)
            embeddings = cached_path('%sglove.gensim' % base_path)

        # KOMNIOS embeddings
        if embeddings.lower() == 'extvec':
            cached_path('%sextvec.gensim.vectors.npy' % base_path)
            embeddings = cached_path('%sextvec.gensim' % base_path)

        # FT-CRAWL embeddings
        if embeddings.lower() == 'ft-crawl':
            cached_path('%sft-crawl.gensim.vectors.npy' % base_path)
            embeddings = cached_path('%sft-crawl.gensim' % base_path)

        # FT-GERMAN embeddings
        if embeddings.lower() == 'ft-german':
            cached_path('%sft-wiki-de.gensim.vectors.npy' % base_path)
            embeddings = cached_path('%sft-wiki-de.gensim' % base_path)

        self.name = embeddings
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(embeddings)

        self.known_words = set(self.precomputed_word_embeddings.index2word)

        self.embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    def embedding_length(self) -> int:
        return self.embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if token.text in self.known_words:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.known_words:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '0', token.text.lower()) in self.known_words:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')

                word_embedding = torch.autograd.Variable(torch.FloatTensor(word_embedding))
                token.set_embedding(self.name, word_embedding)

        return sentences


class CharacterEmbeddings(TextEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(self, path_to_char_dict: str = 'resources/common_characters'):
        """Use the default character dictionary if none provided."""

        super(CharacterEmbeddings, self).__init__()
        self.name = 'Char'
        self.static_embeddings = False

        # load dictionary
        self.char_dictionary: Dictionary = Dictionary()
        with open(path_to_char_dict, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            self.char_dictionary.item2idx = item2idx
            self.char_dictionary.idx2item = idx2item
            # print(self.char_dictionary.item2idx)

        self.char_embedding_dim: int = 25
        self.hidden_size_char: int = 25
        self.char_embedding = torch.nn.Embedding(len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(self.char_embedding_dim, self.hidden_size_char, num_layers=1,
                                      bidirectional=True)

        self.embedding_length = self.char_embedding_dim * 2

    def embedding_length(self) -> int:
        return self.embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                token: Token = token
                # print(token)
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

            tokens_mask = torch.autograd.Variable(torch.LongTensor(tokens_mask))

            # chars for rnn processing
            chars = tokens_mask
            if torch.cuda.is_available():
                chars = chars.cuda()

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(character_embeddings, chars2_length)

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.autograd.Variable(
                torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if torch.cuda.is_available():
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number].cpu())


class CharLMEmbeddings(TextEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self, model, detach: bool = True):
        super().__init__()

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
        """

        # news-english-forward
        if model.lower() == 'news-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward.pt'
            model = cached_path(base_path)

        # news-english-backward
        if model.lower() == 'news-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward.pt'
            model = cached_path(base_path)

        # mix-english-forward
        if model.lower() == 'mix-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward.pt'
            model = cached_path(base_path)

        # mix-english-backward
        if model.lower() == 'mix-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward.pt'
            model = cached_path(base_path)

        # mix-english-forward
        if model.lower() == 'german-forward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward.pt'
            model = cached_path(base_path)

        # mix-english-backward
        if model.lower() == 'german-backward':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward.pt'
            model = cached_path(base_path)

        self.name = model
        self.static_embeddings = detach

        self.lm: RNNModel = RNNModel.load_language_model(model)
        if torch.cuda.is_available():
            self.lm = self.lm.cuda()
        self.lm.eval()

        self.detach = detach

        self.is_forward_lm: bool = self.lm.is_forward_lm
        if self.is_forward_lm:
            print('FORWARD language mode loaded')
        else:
            print('BACKWARD language mode loaded')

        print('on cuda:')
        print(next(self.lm.parameters()).is_cuda)

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed([dummy_sentence])
        self.embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())


    def embedding_length(self) -> int:
        return self.embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # find longest sentence by characters
        longest_character_sequence_in_batch: int = 0
        for sentence in sentences:
            if len(
                    sentence.to_plain_string()) > longest_character_sequence_in_batch: longest_character_sequence_in_batch \
                = len(sentence.to_plain_string())

        sentences_padded: List[str] = []

        for sentence in sentences:
            if self.is_forward_lm:
                sentences_padded.append(
                    '\n' + sentence.to_plain_string() + ' ' + (
                        (longest_character_sequence_in_batch - len(sentence.to_plain_string())) * ' '))
            else:
                sentences_padded.append(
                    '\n' + sentence.to_plain_string()[::-1] + ' ' + (
                        (longest_character_sequence_in_batch - len(sentence.to_plain_string())) * ' '))

        # print(sentences_padded)

        # get states from LM
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

        for i, sentence in enumerate(sentences):

            offset_forward: int = 1
            offset_backward: int = len(sentence.to_plain_string()) + 1

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                offset_forward += len(token.text)

                if self.is_forward_lm:
                    offset = offset_forward
                else:
                    offset = offset_backward

                embedding = all_hidden_states_in_lm[offset, i, :].data.cpu()
                # if not torch.cuda.is_available():
                #     embedding = embedding.cpu()

                offset_forward += 1

                offset_backward -= 1
                offset_backward -= len(token.text)

                token.set_embedding(self.name, torch.autograd.Variable(embedding))
                self.embedding_length = len(embedding)

        return sentences


class OnePassStoreEmbeddings(TextEmbeddings):
    def __init__(self, embedding_stack: StackedEmbeddings, corpus: TaggedCorpus, detach: bool = True):
        super().__init__()

        self.embedding_stack = embedding_stack
        self.detach = detach
        self.name = 'Stack'
        self.static_embeddings = True

        self.embedding_length: int = embedding_stack.embedding_length
        print(self.embedding_length)

        sentences = corpus.get_all_sentences()
        mini_batch_size: int = 32
        sentence_no: int = 0
        written_embeddings: int = 0

        total_count = 0
        for sentence in sentences:
            for token in sentence.tokens:
                total_count += 1

        embeddings_vec = 'fragment_embeddings.vec'
        with open(embeddings_vec, 'a') as f:

            f.write('%d %d\n' % (total_count, self.embedding_stack.embedding_length))

            batches = [sentences[x:x + mini_batch_size] for x in
                       range(0, len(sentences), mini_batch_size)]

            for batch in batches:

                self.embedding_stack.embed(batch)

                for sentence in batch:
                    sentence: Sentence = sentence
                    sentence_no += 1
                    print('%d\t(%d)' % (sentence_no, written_embeddings))
                    # lines: List[str] = []

                    for token in sentence.tokens:
                        token: Token = token

                        signature = self.get_signature(token)
                        vector = token.get_embedding().data.numpy().tolist()
                        vector = ' '.join(map(str, vector))
                        vec = signature + ' ' + vector
                        # lines.append(vec)
                        written_embeddings += 1
                        token.clear_embeddings()

                        f.write('%s\n' % vec)

        vectors = gensim.models.KeyedVectors.load_word2vec_format(embeddings_vec, binary=False)
        vectors.save('stored_embeddings')
        import os
        os.remove('fragment_embeddings.vec')
        vectors = None

        self.embeddings = WordEmbeddings('stored_embeddings')

    def get_signature(self, token: Token) -> str:
        context: str = ' '
        for i in range(token.idx - 4, token.idx + 5):
            if token.sentence.get_token(i) is not None:
                context += token.sentence.get_token(i).text + ' '
        signature = '%s··%d:··%s' % (token.text, token.idx, context)
        return signature.strip().replace(' ', '·')

    def embed(self, sentences: List[Sentence], static_embeddings: bool = True):

        for sentence in sentences:
            for token in sentence.tokens:
                signature = self.get_signature(token)
                word_embedding = self.embeddings.precomputed_word_embeddings.get_vector(signature)
                word_embedding = torch.autograd.Variable(torch.FloatTensor(word_embedding))
                token.set_embedding(self.name, word_embedding)

    def embedding_length(self) -> int:
        return self.embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        return sentences
