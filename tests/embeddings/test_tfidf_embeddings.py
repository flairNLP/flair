from flair.data import Sentence
from flair.embeddings import DocumentTFIDFEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest


class TFIDFEmbeddingsTest(BaseEmbeddingsTest):
    embedding_cls = DocumentTFIDFEmbeddings
    is_document_embedding = True
    is_token_embedding = False

    default_args = {
        "train_dataset": [
            Sentence("This is a sentence"),
            Sentence("This is another sentence"),
            Sentence("another a This I Berlin"),
        ]
    }
