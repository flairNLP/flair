from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import EntityLinker


def test_entity_linker_with_no_candidates():
    linker: EntityLinker = EntityLinker(
        TransformerWordEmbeddings(model="distilbert-base-uncased"), label_dictionary=Dictionary()
    )

    sentence = Sentence("I live in Berlin")
    linker.predict(sentence)


def test_forward_loss():
    sentence = Sentence("I love NYC and hate OYC")
    sentence[2:3].add_label("nel", "New York City")
    sentence[5:6].add_label("nel", "Old York City")

    # init tagger and do a forward pass
    tagger = EntityLinker(TransformerWordEmbeddings("distilbert-base-uncased"), label_dictionary=Dictionary())
    loss, count = tagger.forward_loss([sentence])
    assert count == 2
    assert loss.size() == ()
