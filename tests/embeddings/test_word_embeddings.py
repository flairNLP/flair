from typing import Any, Dict

from flair.embeddings import MuseCrosslingualEmbeddings, NILCEmbeddings, WordEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestWordEmbeddings(BaseEmbeddingsTest):
    embedding_cls = WordEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = dict(embeddings="turian")

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]


class TestMuseCrosslingualEmbeddings(BaseEmbeddingsTest):
    embedding_cls = MuseCrosslingualEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args: Dict[str, Any] = dict()


class TestNILCEmbeddings(BaseEmbeddingsTest):
    embedding_cls = NILCEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = dict(embeddings="glove")
    valid_args = [dict(embeddings="fasttext", model="cbow")]

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]
