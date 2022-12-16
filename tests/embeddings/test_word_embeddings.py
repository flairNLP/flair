from flair.embeddings import WordEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestWordEmbeddings(BaseEmbeddingsTest):
    embedding_cls = WordEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = dict(embeddings="turian")

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]
