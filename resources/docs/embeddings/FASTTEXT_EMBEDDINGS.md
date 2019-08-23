## FastText Embeddings

FastText Embeddings can give you vectors for out of vocabulary(oov) words by using the sub-word information. To use this functionality with Flair, use `FastTextEmbeddings` class as shown:

```python
from flair.embeddings import FastTextEmbeddings

# init embedding
embedding = FastTextEmbeddings('/path/to/local/custom_fasttext_embeddings.bin')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

You can initialize the class by passing the remote downloadable URL as well.

```python
embedding = FastTextEmbeddings('/path/to/remote/downloadable/custom_fasttext_embeddings.bin', use_local=False)
```

