from typing import Any

from flair.embeddings import (
    DocumentCNNEmbeddings,
    DocumentLMEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    FlairEmbeddings,
    TokenEmbeddings,
    WordEmbeddings,
)
from tests.embedding_test_utils import BaseEmbeddingsTest

word: TokenEmbeddings = WordEmbeddings("turian")
flair_embedding: TokenEmbeddings = FlairEmbeddings("news-forward-fast")
flair_embedding_back: TokenEmbeddings = FlairEmbeddings("news-backward-fast")


class BaseDocumentsViaWordEmbeddingsTest(BaseEmbeddingsTest):
    is_document_embedding = True
    is_token_embedding = False
    base_embeddings: list[TokenEmbeddings] = [word, flair_embedding]

    def create_embedding_from_name(self, name: str):
        """Overwrite this method if it is more complex to load an embedding by name."""
        assert self.name_field is not None
        kwargs = dict(self.default_args)
        kwargs.pop(self.name_field)
        return self.embedding_cls(name, **kwargs)  # type: ignore[call-arg]

    def create_embedding_with_args(self, args: dict[str, Any]):
        kwargs = dict(self.default_args)
        for k, v in args.items():
            kwargs[k] = v
        return self.embedding_cls(self.base_embeddings, **kwargs)  # type: ignore[call-arg]


class TestDocumentLstmEmbeddings(BaseDocumentsViaWordEmbeddingsTest):
    embedding_cls = DocumentRNNEmbeddings
    default_args = {
        "hidden_size": 128,
        "bidirectional": False,
    }
    valid_args = [{"bidirectional": False}, {"bidirectional": True}]


class TestDocumentPoolEmbeddings(BaseDocumentsViaWordEmbeddingsTest):
    embedding_cls = DocumentPoolEmbeddings
    default_args = {
        "fine_tune_mode": "nonlinear",
    }
    valid_args = [{"pooling": "mean"}, {"pooling": "max"}, {"pooling": "min"}]


class TestDocumentCNNEmbeddings(BaseDocumentsViaWordEmbeddingsTest):
    embedding_cls = DocumentCNNEmbeddings
    default_args = {
        "kernels": ((50, 2), (50, 3)),
    }
    valid_args = [{"reproject_words_dimension": None}, {"reproject_words_dimension": 100}]


class TestDocumentLMEmbeddings(BaseDocumentsViaWordEmbeddingsTest):
    embedding_cls = DocumentLMEmbeddings
    base_embeddings = [flair_embedding, flair_embedding_back]
    default_args: dict[str, Any] = {}
