import logging

import flair
import numpy as np
from flair.embeddings import WordEmbeddings
import sqlite3
import torch
import re
import os
from tqdm import tqdm
import pickle
import shutil
# this is the default init size of a lmdb database for embeddings
DEFAULT_MAP_SIZE = 100 * 1024 * 1024 * 1024

log = logging.getLogger("flair")

class WordEmbeddingsStore:
    """
    class to simulate a WordEmbeddings class from flair.

    Run this to generate a headless (without word embeddings) model as well a stored word embeddings:

    >>> from flair.inference_utils import WordEmbeddingsStore
    >>> from flair.models import SequenceTagger
    >>> import pickle
    >>> tagger = SequenceTagger.load("multi-ner-fast")
    >>> WordEmbeddingsStore.create_stores(tagger)
    >>> pickle.dump(tagger, open("multi-ner-fast-headless.pickle", "wb"))

    The same but using LMDB as memory database:

    >>> from flair.inference_utils import WordEmbeddingsStore
    >>> from flair.models import SequenceTagger
    >>> import pickle
    >>> tagger = SequenceTagger.load("multi-ner-fast")
    >>> WordEmbeddingsStore.create_stores(tagger, backend='lmdb')
    >>> pickle.dump(tagger, open("multi-ner-fast-headless.pickle", "wb"))

    Then this can be used as follows:

    >>> from flair.data import Sentence
    >>> tagger = pickle.load(open("multi-ner-fast-headless.pickle", "rb"))
    >>> WordEmbeddingsStore.load_stores(tagger)
    >>> text = "Schade um den Ameisenbären. Lukas Bärfuss veröffentlicht Erzählungen aus zwanzig Jahren."
    >>> sentence = Sentence(text)
    >>> tagger.predict(sentence)
    >>> print(sentence.get_spans('ner'))

    The same but using LMDB as memory database:

    >>> from flair.data import Sentence
    >>> tagger = pickle.load(open("multi-ner-fast-headless.pickle", "rb"))
    >>> WordEmbeddingsStore.load_stores(tagger, backend='lmdb')
    >>> text = "Schade um den Ameisenbären. Lukas Bärfuss veröffentlicht Erzählungen aus zwanzig Jahren."
    >>> sentence = Sentence(text)
    >>> tagger.predict(sentence)
    >>> print(sentence.get_spans('ner'))
    """

    def __init__(self, embedding, backend='sqlite', verbose=True):
        """
        :param embedding: Flair WordEmbeddings instance.
        :param backend: cache database backend name e.g ``'sqlite'``, ``'lmdb'``.
                        Default value is ``'sqlite'``.
        :param verbose: If `True` print information on standard output
        """
        # some non-used parameter to allow print
        self._modules = dict()
        self.items = ""

        # get db filename from embedding name
        self.name = embedding.name
        self.store_filename = WordEmbeddingsStore._get_store_filename(embedding, backend)
        if verbose:
            print("store filename:", self.store_filename)

        if backend == 'sqlite':
            self.backend = SqliteWordEmbeddingsStoreBackend(embedding, verbose)
        elif backend == 'lmdb':
            self.backend = LmdbWordEmbeddingsStoreBackend(embedding, verbose)
        else:
            raise ValueError(
                f'The given backend "{backend}" is not available.'
            )

    def _get_vector(self, word="house"):
        return self.backend._get_vector(word)

    def embed(self, sentences):
        for sentence in sentences:
            for token in sentence:
                t = torch.tensor(self._get_vector(word=token.text.lower()))
                token.set_embedding(self.name, t)

    @staticmethod
    def _get_store_filename(embedding, backend='sqlite'):
        """
        get the filename of the store
        """
        embedding_filename = re.findall(".flair(/.*)", embedding.name)[0]
        store_filename = str(flair.cache_root) + embedding_filename + "." + backend
        return store_filename

    @staticmethod
    def create_stores(model, backend='sqlite'):
        """
        creates database versions of all word embeddings in the model and
        deletes the original vectors to save memory
        """
        for embedding in model.embeddings.embeddings:
            if type(embedding) == WordEmbeddings:
                WordEmbeddingsStore(embedding, backend)
                del embedding.precomputed_word_embeddings

    @staticmethod
    def load_stores(model, backend='sqlite'):
        """
        loads the db versions of all word embeddings in the model
        """
        for i, embedding in enumerate(model.embeddings.embeddings):
            if type(embedding) == WordEmbeddings:
                model.embeddings.embeddings[i] = WordEmbeddingsStore(embedding, backend)

    @staticmethod
    def delete_stores(model, backend='sqlite'):
        """
        deletes the db versions of all word embeddings
        """
        for embedding in model.embeddings.embeddings:
            store_filename = WordEmbeddingsStore._get_store_filename(embedding)
            if os.path.isfile(store_filename):
                print("delete store:", store_filename)
                os.remove(store_filename)
            elif os.path.isdir(store_filename):
                print("delete store:", store_filename)
                shutil.rmtree(store_filename, ignore_errors=False, onerror=None)


class WordEmbeddingsStoreBackend:
    def __init__(self, embedding, backend, verbose=True):
        # get db filename from embedding name
        self.name = embedding.name
        self.store_filename = WordEmbeddingsStore._get_store_filename(embedding, backend)
        if verbose:
            print("store filename:", self.store_filename)

    def _get_vector(self, word="house"):
        pass


class SqliteWordEmbeddingsStoreBackend(WordEmbeddingsStoreBackend):
    def __init__(self, embedding, verbose):
        super().__init__(embedding, 'sqlite', verbose)
        # if embedding database already exists
        if os.path.isfile(self.store_filename):
            self.db = sqlite3.connect(self.store_filename)
            cursor = self.db.cursor()
            cursor.execute("SELECT * FROM embedding LIMIT 1;")
            result = list(cursor)
            self.k = len(result[0]) - 1
            return

        # otherwise, push embedding to database
        self.db = sqlite3.connect(self.store_filename)
        pwe = embedding.precomputed_word_embeddings
        self.k = pwe.vector_size
        self.db.execute(f"DROP TABLE IF EXISTS embedding;")
        self.db.execute(
            f"CREATE TABLE embedding(word text,{','.join('v' + str(i) + ' float' for i in range(self.k))});"
        )
        vectors_it = (
            [word] + pwe.get_vector(word).tolist() for word in pwe.vocab.keys()
        )
        if verbose:
            print("load vectors to store")
        self.db.executemany(
            f"INSERT INTO embedding(word,{','.join('v' + str(i) for i in range(self.k))}) \
        values ({','.join(['?'] * (1 + self.k))})",
            tqdm(vectors_it),
        )
        self.db.execute(f"DROP INDEX IF EXISTS embedding_index;")
        self.db.execute(f"CREATE INDEX embedding_index ON embedding(word);")
        self.db.commit()
        self.db.close()

    def _get_vector(self, word="house"):
        db = sqlite3.connect(self.store_filename)
        cursor = db.cursor()
        word = word.replace('"', '')
        cursor.execute(f'SELECT * FROM embedding WHERE word="{word}";')
        result = list(cursor)
        db.close()
        if not result:
            return [0.0] * self.k
        return result[0][1:]


class LmdbWordEmbeddingsStoreBackend(WordEmbeddingsStoreBackend):
    def __init__(self, embedding, verbose):
        super().__init__(embedding, 'lmdb', verbose)
        try:
            import lmdb
            # if embedding database already exists
            load_db = True
            if os.path.isdir(self.store_filename):
                # open the database in read mode
                self.env = lmdb.open(self.store_filename, readonly=True, max_readers=2048, max_spare_txns=4)
                if self.env:
                    # we need to set self.k
                    with self.env.begin() as txn:
                        cursor = txn.cursor()
                        for key, value in cursor:
                            vector = pickle.loads(value)
                            self.k = vector.shape[0]
                            break
                        cursor.close()
                    return
            # create and load the database in write mode
            os.makedirs(self.store_filename, exist_ok=True)
            pwe = embedding.precomputed_word_embeddings
            self.k = pwe.vector_size
            self.env = lmdb.open(self.store_filename, map_size=DEFAULT_MAP_SIZE)
            if verbose:
                print("load vectors to store")
            txn = self.env.begin(write=True)
            for word in tqdm(pwe.vocab.keys()):
                vector = pwe.get_vector(word)
                if len(word.encode(encoding='UTF-8')) < self.env.max_key_size():
                    txn.put(word.encode(encoding='UTF-8'), pickle.dumps(vector))
            txn.commit()
            return
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "lmdb" is not installed!')
            log.warning(
                'To use LMDB, please first install with "pip install lmdb"'
            )
            log.warning("-" * 100)

    def _get_vector(self, word="house"):
        try:
            import lmdb
            with self.env.begin() as txn:
                vector = txn.get(word.encode(encoding='UTF-8'))
                if vector:
                    word_vector = pickle.loads(vector)
                    vector = None
                else:
                    word_vector = np.zeros((self.k,), dtype=np.float32)
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env.close()
            self.env = lmdb.open(self.store_filename, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self._get_vector(word)
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "lmdb" is not installed!')
            log.warning(
                'To use LMDB, please first install with "pip install lmdb"'
            )
            log.warning("-" * 100)
            word_vector = np.zeros((self.k,), dtype=np.float32)
        return word_vector
