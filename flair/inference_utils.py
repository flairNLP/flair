import logging
import pickle
import re
import shutil
import sqlite3
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

import flair
from flair.embeddings import WordEmbeddings

# this is the default init size of a lmdb database for embeddings
DEFAULT_MAP_SIZE = 100 * 1024 * 1024 * 1024

logger = logging.getLogger("flair")

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

    def __init__(self, embedding : WordEmbeddings, backend='sqlite', verbose=True):
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
        self.store_path: Path = WordEmbeddingsStore._get_store_path(embedding, backend)
        if verbose:
            logger.info(f"store filename: {str(self.store_path)}")

        if backend == 'sqlite':
            self.backend = SqliteWordEmbeddingsStoreBackend(embedding, verbose)
        elif backend == 'lmdb':
            self.backend = LmdbWordEmbeddingsStoreBackend(embedding, verbose)
        else:
            raise ValueError(
                f'The given backend "{backend}" is not available.'
            )
        # In case initialization of cached version failed, just fallback to the original WordEmbeddings
        if not self.backend.is_ok:
            self.backend = WordEmbeddings(embedding.embeddings)

    def _get_vector(self, word="house"):
        return self.backend._get_vector(word)

    def embed(self, sentences):
        for sentence in sentences:
            for token in sentence:
                t = torch.tensor(self._get_vector(word=token.text.lower()))
                token.set_embedding(self.name, t)

    @staticmethod
    def _get_store_path(embedding, backend='sqlite'):
        """
        get the filename of the store
        """
        cache_dir = Path(flair.cache_root)
        embedding_filename = re.findall("/(embeddings/.*)", embedding.name)[0]
        store_path = cache_dir / (embedding_filename + "." + backend)
        return store_path

    @staticmethod
    def _word_embeddings(model):
        # SequenceTagger
        if hasattr(model, 'embeddings'):
            embeds = model.embeddings.embeddings
        # TextClassifier
        elif hasattr(model, 'document_embeddings') and hasattr(model.document_embeddings, 'embeddings'):
            embeds = model.document_embeddings.embeddings.embeddings
        else:
            embeds = []
        return embeds

    @staticmethod
    def create_stores(model, backend='sqlite'):
        """
        creates database versions of all word embeddings in the model and
        deletes the original vectors to save memory
        """
        for embedding in WordEmbeddingsStore._word_embeddings(model):
            if type(embedding) == WordEmbeddings:
                WordEmbeddingsStore(embedding, backend)
                del embedding.precomputed_word_embeddings

    @staticmethod
    def load_stores(model, backend='sqlite'):
        """
        loads the db versions of all word embeddings in the model
        """
        embeds = WordEmbeddingsStore._word_embeddings(model)
        for i, embedding in enumerate(embeds):
            if type(embedding) == WordEmbeddings:
                embeds[i] = WordEmbeddingsStore(embedding, backend)

    @staticmethod
    def delete_stores(model, backend='sqlite'):
        """
        deletes the db versions of all word embeddings
        """
        for embedding in WordEmbeddingsStore._word_embeddings(model):
            store_path : Path = WordEmbeddingsStore._get_store_path(embedding)
            logger.info(f"delete store: {str(store_path)}")
            if store_path.is_file():
                store_path.unlink()
            elif store_path.is_dir():
                shutil.rmtree(store_path, ignore_errors=False, onerror=None)


class WordEmbeddingsStoreBackend:
    def __init__(self, embedding, backend, verbose=True):
        # get db filename from embedding name
        self.name = embedding.name
        self.store_path : Path = WordEmbeddingsStore._get_store_path(embedding, backend)

    @property
    def is_ok(self):
        return hasattr(self, 'k')

    def _get_vector(self, word="house"):
        pass


class SqliteWordEmbeddingsStoreBackend(WordEmbeddingsStoreBackend):
    def __init__(self, embedding, verbose):
        super().__init__(embedding, 'sqlite', verbose)
        # if embedding database already exists
        if self.store_path.exists() and self.store_path.is_file():
            try:
                self.db = sqlite3.connect(str(self.store_path))
                cursor = self.db.cursor()
                cursor.execute("SELECT * FROM embedding LIMIT 1;")
                result = list(cursor)
                self.k = len(result[0]) - 1
                return
            except sqlite3.Error as err:
                logger.exception(f"Fail to open sqlite database {str(self.store_path)}: {str(err)}")
        # otherwise, push embedding to database
        if hasattr(embedding, 'precomputed_word_embeddings'):
            self.db = sqlite3.connect(str(self.store_path))
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
                logger.info("load vectors to store")
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
        db = sqlite3.connect(str(self.store_path))
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
            if self.store_path.exists() and self.store_path.is_dir():
                # open the database in read mode
                try:
                    self.env = lmdb.open(str(self.store_path), readonly=True, max_readers=2048, max_spare_txns=4)
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
                except lmdb.Error as err:
                    logger.exception(f"Fail to open lmdb database {str(self.store_path)}: {str(err)}")
            # create and load the database in write mode
            if hasattr(embedding, 'precomputed_word_embeddings'):
                pwe = embedding.precomputed_word_embeddings
                self.k = pwe.vector_size
                self.store_path.mkdir(parents=True, exist_ok=True)
                self.env = lmdb.open(str(self.store_path), map_size=DEFAULT_MAP_SIZE)
                if verbose:
                    logger.info("load vectors to store")

                txn = self.env.begin(write=True)
                for word in tqdm(pwe.vocab.keys()):
                    vector = pwe.get_vector(word)
                    if len(word.encode(encoding='UTF-8')) < self.env.max_key_size():
                        txn.put(word.encode(encoding='UTF-8'), pickle.dumps(vector))
                txn.commit()
                return
        except ModuleNotFoundError:
            logger.warning("-" * 100)
            logger.warning('ATTENTION! The library "lmdb" is not installed!')
            logger.warning(
                'To use LMDB, please first install with "pip install lmdb"'
            )
            logger.warning("-" * 100)

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
            self.env = lmdb.open(self.store_path, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self._get_vector(word)
        except ModuleNotFoundError:
            logger.warning("-" * 100)
            logger.warning('ATTENTION! The library "lmdb" is not installed!')
            logger.warning(
                'To use LMDB, please first install with "pip install lmdb"'
            )
            logger.warning("-" * 100)
            word_vector = np.zeros((self.k,), dtype=np.float32)
        return word_vector
