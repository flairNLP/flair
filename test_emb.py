from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

phrase_0 = Sentence("a  uui")
embeddings_a = TransformerWordEmbeddings(
    'roberta-base',
    use_context=True,
    use_context_separator=False,
)
ebd_a = embeddings_a.embed(phrase_0)

phrase_1 = Sentence("a  uui")
embeddings_b = TransformerWordEmbeddings(
    'roberta-base',
    use_context=True,
    use_context_separator=False,
)
ebd_b = embeddings_b.embed(phrase_1)
ebd_b = [phrase_1]
ebd_a = [phrase_0]

print(
    "token run 0:", ebd_a[-1][-1], "\n",
    "embedding end run 0:", ebd_a[-1][-1].embedding.tolist()[-2:], "\n",
    "token run 1: ", ebd_b[-1][-1], "\n",
    "embedding end run 1:", ebd_b[-1][-1].embedding.tolist()[-2:]
)