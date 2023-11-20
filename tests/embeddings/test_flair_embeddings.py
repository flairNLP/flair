from flair.data import Dictionary, Sentence
from flair.embeddings import (
    DocumentLMEmbeddings,
    DocumentRNNEmbeddings,
    FlairEmbeddings,
)
from flair.models import LanguageModel
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestFlairEmbeddings(BaseEmbeddingsTest):
    embedding_cls = FlairEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"model": "news-forward-fast"}

    name_field = "model"
    invalid_names = ["other", "not/existing/path/to/embeddings"]

    def test_fine_tunable_flair_embedding(self):
        language_model_forward = LanguageModel(Dictionary.load("chars"), is_forward_lm=True, hidden_size=32, nlayers=1)

        embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
            [FlairEmbeddings(language_model_forward, fine_tune=True)],
            hidden_size=128,
            bidirectional=False,
        )

        sentence: Sentence = Sentence("I love Berlin.")

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 128
        assert len(sentence.get_embedding()) == embeddings.embedding_length

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0

        embeddings: DocumentLMEmbeddings = DocumentLMEmbeddings(
            [FlairEmbeddings(language_model_forward, fine_tune=True)]
        )

        sentence: Sentence = Sentence("I love Berlin.")

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 32
        assert len(sentence.get_embedding()) == embeddings.embedding_length

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings
