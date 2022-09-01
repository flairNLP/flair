from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

model = SequenceTagger.load("ner-large")
assert isinstance(model.embeddings, (TransformerWordEmbeddings, TransformerDocumentEmbeddings))

from flair.data import Sentence
sentences = [Sentence("Hi Pier, nice to meet you"), Sentence("This is the first positive result!")]

print(model.predict(sentences))

model.embeddings = model.embeddings.optimize_nebuly(sentences, ignore_compilers=["tvm"])

print(model.predict(sentences))
