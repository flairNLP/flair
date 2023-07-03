# Other embeddings in Flair

Flair supports many other embedding types. This section introduces these embeddings.

```{note}
We mostly train our models with either [`TransformerEmbeddings`](#flair.embeddings.transformer.TransformerEmbeddings) or [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings). The embeddings presented here might be useful 
for specific use cases or for comparison purposes. 
```


## One-Hot Embeddings

[`OneHotEmbeddings`](#flair.embeddings.token.OneHotEmbeddings) are embeddings that encode each word in a vocabulary as a one-hot vector, followed by an embedding
layer. These embeddings
thus do not encode any prior knowledge as do most other embeddings. They also differ in that they
require to see a vocabulary (`vocab_dictionary`) during instantiation. Such dictionary can be passed as an argument
during class initialization or constructed directly from a corpus with a [`OneHotEmbeddings.from_corpus`](#flair.embeddings.token.OneHotEmbeddings.from_corpus) method. The dictionary consists
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
of 300. However, this layer is randomly initialized, meaning that these embeddings do not make sense unless they are trained in a task.

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


### Embedding other tags

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


## Byte Pair Embeddings

[`BytePairEmbeddings`](#flair.embeddings.token.BytePairEmbeddings) are word embeddings that are precomputed on the subword-level. This means that they are able to
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

[`BytePairEmbeddings`](#flair.embeddings.token.BytePairEmbeddings) also have a multilingual model capable of embedding any word in any language.
 You can instantiate it with:

```python
# init embedding
embedding = BytePairEmbeddings('multi')
```

You can also load custom [`BytePairEmbeddings`](#flair.embeddings.token.BytePairEmbeddings) by specifying a path to model_file_path and embedding_file_path arguments. They correspond respectively to a SentencePiece model file and to an embedding file (Word2Vec plain text or GenSim binary). For example:

```python
# init custom embedding
embedding = BytePairEmbeddings(model_file_path='your/path/m.model', embedding_file_path='your/path/w2v.txt')
```

## Document Pool Embeddings

[`DocumentPoolEmbeddings`](#flair.embeddings.document.DocumentPoolEmbeddings) calculate a pooling operation over all word embeddings in a document.
The default operation is `mean` which gives us the mean of all words in the sentence.
The resulting embedding is taken as document embedding.

To create a mean document embedding simply create any number of [`TokenEmbeddings`](#flair.embeddings.base.TokenEmbeddings) first and put them in a list.
Afterwards, initiate the [`DocumentPoolEmbeddings`](#flair.embeddings.document.DocumentPoolEmbeddings) with this list of [`TokenEmbeddings`](#flair.embeddings.base.TokenEmbeddings).
So, if you want to create a document embedding using GloVe embeddings together with [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings),
use the following code:

```python
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding])
```

Now, create an example sentence and call the embedding's [`embed()`](#flair.embeddings.base.Embeddings.embed) method.

```python
# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.embedding)
```

This prints out the embedding of the document. Since the document embedding is derived from word embeddings, its dimensionality depends on the dimensionality of word embeddings you are using.

You have the following optional constructor arguments:

| Argument             | Default             | Description
| -------------------- | ------------------- | ------------------------------------------------------------------------------
| `fine_tune_mode`             | `linear`       | One of `linear`, `nonlinear` and `none`.
| `pooling`  | `first`             | One of `mean`, `max` and `min`.

### Pooling operation

Next to the `mean` pooling operation you can also use `min` or `max` pooling. Simply pass the pooling operation you want
to use to the initialization of the `DocumentPoolEmbeddings`:
```python
document_embeddings = DocumentPoolEmbeddings([glove_embedding],  pooling='min')
```

### Fine-tune mode

You can also choose which fine-tuning operation you want, i.e. which transformation to apply before word embeddings get
pooled. The default operation is 'linear' transformation, but if you only use simple word embeddings that are
not task-trained you should probably use a 'nonlinear' transformation instead:

```python
# instantiate pre-trained word embeddings
embeddings = WordEmbeddings('glove')

# document pool embeddings
document_embeddings = DocumentPoolEmbeddings([embeddings], fine_tune_mode='nonlinear')
```

If on the other hand you use word embeddings that are task-trained (such as simple one hot encoded embeddings), you
are often better off doing no transformation at all. Do this by passing 'none':

```python
# instantiate one-hot encoded word embeddings
embeddings = OneHotEmbeddings(corpus)

# document pool embeddings
document_embeddings = DocumentPoolEmbeddings([embeddings], fine_tune_mode='none')
```

## Document RNN Embeddings

Besides simple pooling we also support a method based on an RNN to obtain a [`DocumentEmbeddings`](#flair.embeddings.base.DocumentEmbeddings).
The RNN takes the word embeddings of every token in the document as input and provides its last output state as document
embedding. You can choose which type of RNN you wish to use.

In order to use the [`DocumentRNNEmbeddings`](#flair.embeddings.document.DocumentRNNEmbeddings) you need to initialize them by passing a list of token embeddings to it:

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentRNNEmbeddings([glove_embedding])
```

By default, a GRU-type RNN is instantiated. Now, create an example sentence and call the embedding's [`embed()`](#flair.embeddings.base.Embeddings.embed) method.

```python
# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())
```

This will output a single embedding for the complete sentence. The embedding dimensionality depends on the number of
hidden states you are using and whether the RNN is bidirectional or not.

### RNN type

If you want to use a different type of RNN, you need to set the `rnn_type` parameter in the constructor. So,
to initialize a document RNN embedding with an LSTM, do:

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_lstm_embeddings = DocumentRNNEmbeddings([glove_embedding], rnn_type='LSTM')
```

### Need to be trained on a task

Note that while [`DocumentPoolEmbeddings`](#flair.embeddings.document.DocumentPoolEmbeddings) are immediately meaningful, [`DocumentRNNEmbeddings`](#flair.embeddings.document.DocumentRNNEmbeddings) need to be tuned on the
downstream task. This happens automatically in Flair if you train a new model with these embeddings. 

Once the model is trained, you can access the tuned [`DocumentRNNEmbeddings`](#flair.embeddings.document.DocumentRNNEmbeddings) object directly from the classifier object and use it to embed sentences.

```python
document_embeddings = classifier.document_embeddings

sentence = Sentence('The grass is green . And the sky is blue .')

document_embeddings.embed(sentence)

print(sentence.get_embedding())
```

[`DocumentRNNEmbeddings`](#flair.embeddings.document.DocumentRNNEmbeddings) have a number of hyperparameters that can be tuned, please take a look at their [API docs](#flair.embeddings.document.DocumentRNNEmbeddings) to find out more.
