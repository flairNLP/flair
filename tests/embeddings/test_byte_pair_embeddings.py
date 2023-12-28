from flair.embeddings import BytePairEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestBytePairEmbeddings(BaseEmbeddingsTest):
    embedding_cls = BytePairEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"language": "en"}
