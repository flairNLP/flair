from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import EntityLinker


def test_entity_linker_with_no_candidates():
    linker: EntityLinker = EntityLinker(
        TransformerWordEmbeddings(model="distilbert-base-uncased"), label_dictionary=Dictionary()
    )

    sentence = Sentence("I live in Berlin")
    linker.predict(sentence)
