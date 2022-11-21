from flair.embeddings import TransformerDocumentEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestTransformerDocumentEmbeddings(BaseEmbeddingsTest):
    embedding_cls = TransformerDocumentEmbeddings
    is_document_embedding = True
    is_token_embedding = False
    default_args = dict(model="distilbert-base-uncased", allow_long_sentences=False)
    valid_args = [
        dict(layers="-1,-2,-3,-4", layer_mean=False),
        dict(layers="all", layer_mean=True),
        dict(layers="all", layer_mean=False),
    ]

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]
