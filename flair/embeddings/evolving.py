import re
import unicodedata
import numpy as np
import torch
from collections import Counter
from typing import List, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings


def _preprocess_token(text, to_lower=True, norm_num=True, num="0"):
    # no non-ascii characters
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = nfkd_form.encode("ASCII", "ignore").decode("ASCII")
    if to_lower:
        # all lower case
        text = text.lower()
    if norm_num:
        # normalize numbers
        text = re.sub(r"\d", num, text)
    return text.strip()


def _generate_preprocessed_tokens(text):
    yield text                                    # original text
    t = _preprocess_token(text, False, False)     # without weird characters
    yield t
    yield t.lower()                               # lower case
    yield t.title()                               # title case
    if not t.isalpha():                           # different number normalizations
        yield re.sub(r"\d", "0", t.lower())
        yield re.sub(r"\d", "#", t.lower())
        yield re.sub(r"\d", "0", t)
        yield re.sub(r"\d", "#", t)


def _get_sentence_embeddings_flair(embedding, sentence):
    # transform list of words into flair sentence object (with custom tokenizer to make sure the tokens match!)
    s = Sentence("\t".join(sentence), use_tokenizer=lambda x: x.split("\t"))
    # embed words in sentence
    embedding.embed(s)
    # extract and return embedding matrix
    return np.vstack([t.embedding.cpu().numpy() for t in s])


class EmbeddingInputModel:

    def __init__(self, sentences, min_freq=1, n_tokens=None, verbose=1):
        """
        Initialize the EmbeddingInputModel by letting it check the sentences to generate
        a mapping from token to index and vice versa (used e.g. to index the all the embeddings here)

        :param sentences: a list of lists of words
        :param min_freq: how often a token needs to occur to be considered as a feature
        :param n_tokens: how many tokens to keep at most (might be less depending on min_freq; default: all)
        :param verbose: whether to generate warnings (default: 1)
        """
        self.min_freq = min_freq
        self.token_counts = Counter(t for sentence in sentences for t in sentence)
        self.index2token = [t for t, c in self.token_counts.most_common(n_tokens) if c >= min_freq]
        if not self.index2token and verbose:
            print("[EmbeddingInputModel] WARNING: no tokens with frequency >= %i" % min_freq)
        self.token2index = {t: i for i, t in enumerate(self.index2token)}

    @property
    def n_tokens(self) -> int:
        return len(self.index2token)

    def update_index(self, sentence):
        # possibly add the tokens from the new sentence to the index
        for token in set(sentence):
            if token not in self.token2index:
                self.token2index[token] = len(self.index2token)
                self.index2token.append(token)

    def get_token(self, token_text, default=None):
        # get the closest matching token in our vocab
        if token_text is None:
            return default
        for t in _generate_preprocessed_tokens(token_text):
            if t in self.token2index:
                return t
        return default

    def get_index(self, token_text, default=-1):
        # get the index of the closest matching token in our vocab
        if token_text is None:
            return default
        for t in _generate_preprocessed_tokens(token_text):
            if t in self.token2index:
                return self.token2index[t]
        return default


class PretrainedEmbeddings(TokenEmbeddings):

    def __init__(self, embeddings: Union[KeyedVectors, np.array], input_model: Optional[EmbeddingInputModel] = None, include_oov=True):
        """
        Everything that is needed to embed words with the given pretrained embeddings.

        :param embeddings: either pretrained gensim embeddings or a matrix with n_tokens x embedding_dim
        :param input_model: input_model model (needed if only matrix is passed, else extracted from gensim embeddings)
        :param include_oov: whether an OOV embedding at index -1 needs to be created (default: True; regular gensim embeddings don't come with that)
        """
        super().__init__()
        self.name = "Pretrained"
        self.static_embeddings = True
        # are we dealing with KeyedVectors? then construct our own input model
        if hasattr(embeddings, "index2entity"):
            self.input_model = EmbeddingInputModel([], verbose=0)
            self.input_model.index2token = embeddings.index2entity
            self.input_model.token2index = {t: i for i, t in enumerate(self.input_model.index2token)}
            embeddings = embeddings.vectors
        elif input_model is None:
            raise RuntimeError("[PretrainedEmbeddings] Need either KeyedVectors as embeddings or a EmbeddingInputModel")
        else:
            self.input_model = input_model
        # possibly add OOV embedding at -1
        if include_oov:
            self.embeddings = np.vstack([embeddings, np.zeros(embeddings.shape[1])])
        else:
            self.embeddings = embeddings

    @property
    def embedding_length(self) -> int:
        return self.embeddings.shape[1]

    def __contains__(self, token):
        return self.input_model.get_token(token) is not None

    def __getitem__(self, token):
        # get embedding for a single token (text) as a numpy array with -1 = OOV
        return self.embeddings[self.input_model.get_index(token)]

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sentence in sentences:
            for token in sentence:
                token.set_embedding(self.name, torch.tensor(self.__getitem__(token.text), dtype=torch.float32))
        return sentences

    def get_nneighbors(self, token, k=5, include_simscore=True):
        """
        Inputs:
            - token: token for which to compute the nearest neighbors
            - k: how many nearest neighbors to find (default: 5)
            - include_simscore: whether the similarity score of the token should be included in the results (default: True)
        Returns:
            - nearest_neighbors: list of k tokens that are most similar to the given token (+ similarity score)
        """
        if self.input_model.get_token(token) is None:
            print("not in vocabulary:", token)
            return []
        # nearest neighbors idx based on cosine similarity of token to other embeddings
        sims = cosine_similarity(self.embeddings, self.__getitem__(token).reshape(1, -1)).flatten()
        nn_idx = np.argsort(sims)[::-1]
        # make sure it's not the token itself or OOV
        nn_tokens = []
        for i in nn_idx:
            if i < self.embeddings.shape[0] - 1:
                t = self.input_model.index2token[i]
                if t != token:
                    nn_tokens.append((t, sims[i]) if include_simscore else t)
            if len(nn_tokens) >= k:
                break
        return nn_tokens


class GlobalAvgEmbeddings(TokenEmbeddings):

    def __init__(self, local_embeddings, sentences, min_freq=1, n_tokens=None):
        """
        construct global average of local context embeddings from a given corpus

        :param local_embeddings: embeddings that can be used to get local embeddings for a given sentence
        :param sentences: a list of lists of words over which the embeddings are computed
        :param min_freq: how often a token needs to occur to build an embedding for it (default: 1)
        :param n_tokens: how many tokens to keep at most (might be less depending on min_freq; default: all)
        """
        super().__init__()
        self.name = "GlobalAvg"
        self.static_embeddings = True
        # input model to map from token to embedding
        self.input_model = EmbeddingInputModel(sentences, min_freq, n_tokens)
        # matrix where final embeddings are stored; at index -1 we have the OOV embedding
        self.embeddings = np.zeros((self.input_model.n_tokens+1, local_embeddings.embedding_length))
        # counter for each token (to get streaming averages right)
        self.token_counter = np.zeros(self.input_model.n_tokens+1, dtype=int)
        print("[GlobalAvgEmbeddings] building global context embeddings from sentences (size: %i x %i)" % self.embeddings.shape)
        for n, sentence in enumerate(sentences):
            if not n % 100:
                print("[GlobalAvgEmbeddings] PROGRESS: at sentence %i / %i" % (n, len(sentences)), end="\r")
            # get local embeddings for all words in the sentence
            context_embeddings = _get_sentence_embeddings_flair(local_embeddings, sentence)
            # update global embeddings - for some reason this is faster than np.add.at(self.embeddings, [... vidx ...], context_embeddings)
            for i, vidx in enumerate(self.input_model.get_index(t) for t in sentence):
                # update sum and counter
                self.embeddings[vidx, :] += context_embeddings[i]
                self.token_counter[vidx] += 1
        self.embeddings = self.embeddings/np.maximum(self.token_counter, 1)[:, None]
        print("[GlobalAvgEmbeddings] PROGRESS: at sentence %i / %i" % (len(sentences), len(sentences)))

    @property
    def embedding_length(self) -> int:
        return self.embeddings.shape[1]

    def __contains__(self, token):
        return self.input_model.get_token(token) is not None

    def __getitem__(self, token):
        # get embedding for a single token (text) as a numpy array
        # if get_index returns -1, we get the OOV embedding
        return self.embeddings[self.input_model.get_index(token)]

    def as_pretrained(self):
        return PretrainedEmbeddings(self.embeddings.copy(), self.input_model, include_oov=False)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sentence in sentences:
            for token in sentence:
                token.set_embedding(self.name, torch.tensor(self.__getitem__(token.text), dtype=torch.float32))
        return sentences


class EvolvingEmbeddings(TokenEmbeddings):

    def __init__(self, local_embeddings, sentences=[], alpha=None, reset_token=None, min_freq=1, n_tokens=None, update_index=True):
        """
        construct (weighted) running average of local context embeddings from a given corpus

        :param local_embeddings: embeddings that can be used to get local (i.e. contextualized) embeddings for a given sentence
        :param sentences: list of lists of words; for greater efficiency (if reset_token is None), the input model can already
                          be instantiated for the whole vocabulary in the beginning, otherwise it is updated as new sentences are processed
        :param alpha: influence of the current local embedding on the average (float between 0 and 0.5 or None; default: None)
                      --> if None, it's just a regular running average, otherwise weighted, i.e. embedding = (1-alpha)*avg_emb + alpha*local_emb
        :param reset_token: a certain token, e.g. indicating a new document, that, if it occurs as the first (or only) token in a sentence
                            triggers a reset of the running averages (default: None, i.e., no resets)
        :param min_freq: how often a token needs to occur to build an embedding for it (default: 1; only used when sentences is given and update_index=False)
        :param n_tokens: how many tokens to keep at most (might be less depending on min_freq; default: all; only used when sentences is given and update_index=False)
        :param update_index: whether embeddings for previously unseen tokens should be created (default: True)
                             Use update_index=True (and min_freq=1) to create continuously evolving embeddings for tasks like NER;
                             to get a fixed set of embeddings for a certain vocab, set update_index=False pass sentences + min_freq=x (e.g. for diachronic emb.).
        """
        super().__init__()
        self.static_embeddings = True
        self.local_embeddings = local_embeddings
        if alpha is not None:
            assert alpha > 0., "[EvolvingEmbeddings] alpha needs to be > 0 for evolving to make sense"
            assert alpha <= 0.5, "[EvolvingEmbeddings] alpha needs to be <= 0.5 (or you can just use the local_embeddings by themselves)"
        self.alpha = alpha
        self.reset_token = reset_token
        if reset_token is not None:
            self.name = "EvolveDoc%s" % str(alpha)
        else:
            self.name = "Evolve%s" % str(alpha)
        self.update_index = update_index
        self._reset_embeddings(sentences, min_freq, n_tokens)
        self._set_max_count()

    def extra_repr(self) -> str:
        return "alpha={}".format(self.alpha)

    @property
    def embedding_length(self) -> int:
        return self.embeddings.shape[1]

    def _reset_embeddings(self, sentences=[], min_freq=1, n_tokens=None):
        # input model to map from token to embedding
        self.input_model = EmbeddingInputModel(sentences, min_freq, n_tokens, verbose=0)
        # matrix where final embeddings are stored; at index -1 we have the OOV embedding
        self.embeddings = np.zeros((self.input_model.n_tokens+1, self.local_embeddings.embedding_length))
        # counter for each token (to get running averages right)
        self.token_counter = np.zeros(self.input_model.n_tokens+1, dtype=int)

    def _set_max_count(self, count_dict=None):
        if self.alpha is not None:
            # alpha determines the maximum number of occurrences of a token that are considered for the current average
            # by setting the max_count to 1/alpha - 1 such that (max_count*global + local)/(max_count + 1)
            # is the same as (1-alpha)*global + alpha*local  (unless alpha > 0.5; global will always count for at least half!)
            max_count = int(np.ceil(1/self.alpha)) - 1
        else:
            max_count = np.inf
        if count_dict is not None:
            # create a max_count array from the given values; e.g. count_dict could be input_model.token_counts / number_of_years
            # and don't forget the entry for the OOV tokens
            self.max_count = np.array([count_dict.get(t, max_count) for t in self.input_model.index2token] + [max_count])
        else:
            # just use the single number
            self.max_count = max_count

    def as_pretrained(self):
        return PretrainedEmbeddings(self.embeddings.copy(), self.input_model, include_oov=False)

    def update_evolving_embeddings(self, sentence):
        """
        Update running average embeddings for a single sentence.

        :param sentence: a list of words
        """
        # possibly reset the embeddings
        if self.reset_token is not None and sentence[0] == self.reset_token:
            self._reset_embeddings([sentence])
        elif self.update_index:
            # possibly add new words to the index (and extend the matrix + count vector accordingly...)
            idx_len_before = self.input_model.n_tokens
            self.input_model.update_index(sentence)
            new_entries = self.input_model.n_tokens - idx_len_before
            if new_entries:
                # be careful: -1 corresponds to the OOV embedding!
                self.embeddings = np.vstack([self.embeddings[:-1], np.zeros((new_entries, self.local_embeddings.embedding_length)), self.embeddings[[-1]]])
                self.token_counter = np.hstack([self.token_counter[:-1], np.zeros(new_entries, dtype=int), self.token_counter[-1]])
        # possibly reduce the counter to max_count to compute the alpha-weighted average
        self.token_counter = np.minimum(self.token_counter, self.max_count)
        # get contextualized embeddings for all words in the sentence
        context_embeddings = _get_sentence_embeddings_flair(self.local_embeddings, sentence)
        # update evolving embeddings - for some reason this is faster than np.add.at(self.embeddings, [... vidx ...], context_embeddings)
        for i, vidx in enumerate(self.input_model.get_index(t) for t in sentence):
            self.embeddings[vidx, :] = (self.token_counter[vidx] * self.embeddings[vidx, :] + context_embeddings[i]) / (self.token_counter[vidx] + 1)
            # increase counter
            self.token_counter[vidx] += 1

    def __contains__(self, token):
        return self.input_model.get_token(token) is not None

    def __getitem__(self, token):
        # get embedding for a single token (text) as a numpy array
        # if get_index returns -1, we get the OOV embedding
        return self.embeddings[self.input_model.get_index(token)]

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sentence in sentences:
            # first update
            self.update_evolving_embeddings([t.text for t in sentence if t.text])
            for token in sentence:
                # then get the current embedding (transforming the numpy array to a tensor also ensures it's not just a reference that would get changed in the next update!)
                token.set_embedding(self.name, torch.tensor(self.__getitem__(token.text), dtype=torch.float32))
        return sentences
