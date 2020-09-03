# Byte Pair Embeddings

`BytePairEmbeddings` are word embeddings that are precomputed on the subword-level. This means that they are able to
embed any word by splitting words into subwords and looking up their embeddings. `BytePairEmbeddings` were proposed
and computed by [Heinzerling and Strube (2018)](https://www.aclweb.org/anthology/L18-1473) who found that they offer nearly the same accuracy as word embeddings, but at a fraction
of the model size. So they are a great choice if you want to train small models.

You initialize with a language code (275 languages supported), a number of 'syllables' (one of ) and
a number of dimensions (one of 50, 100, 200 or 300). The following initializes and uses byte pair embeddings
for English:

```python
from flair.embeddings import BytePairEmbeddings

# init embedding
embedding = BytePairEmbeddings('en')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

More information can be found
on the [byte pair embeddings](https://nlp.h-its.org/bpemb/) web page.

`BytePairEmbeddings` also have a multilingual model capable of embedding any word in any language.
 You can instantiate it with:

```python
# init embedding
embedding = BytePairEmbeddings('multi')
```

You can also load custom `BytePairEmbeddings` by specifying a path to model_file_path and embedding_file_path arguments. They correspond respectively to a SentencePiece model file and to an embedding file (Word2Vec plain text or GenSim binary). For example:

```python
# init custom embedding
embedding = BytePairEmbeddings(model_file_path='your/path/m.model', embedding_file_path='your/path/w2v.txt')
```
