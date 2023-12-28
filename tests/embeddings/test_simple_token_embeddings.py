from flair.data import Dictionary
from flair.embeddings import CharacterEmbeddings, HashEmbeddings, OneHotEmbeddings
from tests.embedding_test_utils import BaseEmbeddingsTest

vocab_dictionary = Dictionary(add_unk=True)
vocab_dictionary.add_item("I")
vocab_dictionary.add_item("love")
vocab_dictionary.add_item("berlin")


class TestCharacterEmbeddings(BaseEmbeddingsTest):
    embedding_cls = CharacterEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"path_to_char_dict": None}


class TestOneHotEmbeddings(BaseEmbeddingsTest):
    embedding_cls = OneHotEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"vocab_dictionary": vocab_dictionary}


class TestHashEmbeddings(BaseEmbeddingsTest):
    embedding_cls = HashEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"num_embeddings": 10}
