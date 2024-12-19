from typing import Any

from flair.embeddings import MuseCrosslingualEmbeddings, NILCEmbeddings, WordEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestWordEmbeddings(BaseEmbeddingsTest):
    embedding_cls = WordEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"embeddings": "turian"}

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]


class TestMuseCrosslingualEmbeddings(BaseEmbeddingsTest):
    embedding_cls = MuseCrosslingualEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args: dict[str, Any] = {}


class TestNILCEmbeddings(BaseEmbeddingsTest):
    embedding_cls = NILCEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"embeddings": "fasttext", "model": "cbow", "size": 50}
    valid_args = [{"embeddings": "glove"}]

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]
