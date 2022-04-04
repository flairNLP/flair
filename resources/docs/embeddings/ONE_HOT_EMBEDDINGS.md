# One-Hot Embeddings

`OneHotEmbeddings` are embeddings that encode each word in a vocabulary as a one-hot vector, followed by an embedding 
layer. These embeddings
thus do not encode any prior knowledge as do most other embeddings. They also differ in that they 
require to see a vocabulary (`vocab_dictionary`) during instantiation. Such dictionary can be passed as an argument
during class initialization or constructed directly from a corpus with a `from_corpus` method. The dictionary consists 
of all unique tokens contained in the corpus plus an UNK token for all rare words. 

You initialize these embeddings like this:

```python
from flair.embeddings import OneHotEmbeddings
from flair.datasets import UD_ENGLISH
from flair.data import Sentence

# load a corpus
corpus = UD_ENGLISH()

# init embedding
embeddings = OneHotEmbeddings.from_corpus(corpus)

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embeddings.embed(sentence)
```

By default, the 'text' of a token (i.e. its lexical value) is one-hot encoded and the embedding layer has a dimensionality
of 300. However, this layer is randomly initialized, meaning that these embeddings do not make sense unless they are 
 [trained in a task](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md). 

### Vocabulary size

By default, all words that occur in the corpus at least 3 times are part of the vocabulary. You can change 
this using the `min_freq` parameter. For instance, if your corpus is very large you might want to set a 
higher `min_freq`: 

```python
embeddings = OneHotEmbeddings.from_corpus(corpus, min_freq=10)
```

### Embedding dimensionality

By default, the embeddings have a dimensionality of 300. If you want to try higher or lower values, you can use the 
`embedding_length` parameter:

```python
embeddings = OneHotEmbeddings.from_corpus(corpus, embedding_length=100)
```


## Embedding other tags

Sometimes, you want to embed something other than text. For instance, sometimes we have part-of-speech tags or 
named entity annotation available that we might want to use. If this field exists in your corpus, you can embed
it by passing the field variable. For instance, the UD corpora have a universal part-of-speech tag for each 
token ('upos'). Embed it like so: 

```python
from flair.datasets import UD_ENGLISH
from flair.embeddings import OneHotEmbeddings

# load corpus
corpus = UD_ENGLISH()

# embed POS tags
embeddings = OneHotEmbeddings.from_corpus(corpus, field='upos')
```

This should print a vocabulary of size 18 consisting of universal part-of-speech tags. 
