from flair.data import Dictionary
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.nn import Classifier
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestTransformerDocumentEmbeddings(BaseEmbeddingsTest):
    embedding_cls = TransformerDocumentEmbeddings
    is_document_embedding = True
    is_token_embedding = False
    default_args = {"model": "distilbert-base-uncased", "allow_long_sentences": False}
    valid_args = [
        {"layers": "-1,-2,-3,-4", "layer_mean": False},
        {"layers": "all", "layer_mean": True},
        {"layers": "all", "layer_mean": False},
    ]

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]


def test_if_loaded_embeddings_have_all_attributes(tasks_base_path):
    # dummy model with embeddings
    embeddings = TransformerDocumentEmbeddings(
        "distilbert-base-uncased",
        use_context=True,
        use_context_separator=False,
    )

    model = TextClassifier(label_type="ner", label_dictionary=Dictionary(), embeddings=embeddings)

    # save the dummy and load it again
    model.save(tasks_base_path / "single.pt")
    loaded_single_task = Classifier.load(tasks_base_path / "single.pt")

    # check that context_length and use_context_separator is the same for both
    assert model.embeddings.context_length == loaded_single_task.embeddings.context_length
    assert model.embeddings.use_context_separator == loaded_single_task.embeddings.use_context_separator
