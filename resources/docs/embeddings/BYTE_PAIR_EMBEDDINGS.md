## New: Byte Pair Embeddings

We now also include the byte pair embeddings calulated by @bheinzerling that segment words into subsequences.
This can dramatically reduce the model size vis-a-vis using normal word embeddings at nearly the same accuracy.
So, if you want to train small models try out the new `BytePairEmbeddings` class.

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
Given its memory advantages, we would be interested to hear from the community how well these embeddings work.