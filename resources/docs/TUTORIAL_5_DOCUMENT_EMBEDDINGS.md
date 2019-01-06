# Tutorial 5: Document Embeddings

Document embeddings are different from [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) in that they 
give you one embedding for an entire text, whereas word embeddings give you embeddings for individual words. 

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work.

## Embeddings

All document embedding classes inherit from the `DocumentEmbeddings` class and implement the `embed()` method which you
need to call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are Pytorch vectors, so they can be immediately used for training and
fine-tuning.

## Document Embeddings

Our document embeddings are created from the embeddings of all words in the document.
Currently, we have two different methods to obtain a document embedding from a list of word embeddings.

### Pooling

The first method calculates a pooling operation over all word embeddings in a document.
The default operation is `mean` which gives us the mean of all words in the sentence.
The resulting embedding is taken as document embedding.

To create a mean document embedding simply create any number of `TokenEmbeddings` first and put them in a list.
Afterwards, initiate the `DocumentPoolEmbeddings` with this list of `TokenEmbeddings`.
So, if you want to create a document embedding using GloVe embeddings together with CharLMEmbeddings,
use the following code:

```python
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentPoolEmbeddings, Sentence

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')

# initialize the document embeddings
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              charlm_embedding_backward,
                                              charlm_embedding_forward])
```

Now, create an example sentence and call the embedding's `embed()` method.

```python
# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())
```

This prints out the embedding of the document.
Since the document embedding is derived from word embeddings, its dimensionality depends on the dimensionality of word
embeddings you are using.

Next to the `mean` pooling operation you can also use `min` or `max` pooling. Simply pass the pooling operation you want
to use to the initialization of the `DocumentPoolEmbeddings`:
```python
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              charlm_embedding_backward,
                                              charlm_embedding_forward],
                                             mode='min')
```


### LSTM

Besides the pooling we also support a method based on an LSTM to obtain a `DocumentEmbeddings`.
The LSTM takes the word embeddings of every token in the document as input and provides its last output state as document
embedding.

In order to use the `DocumentLSTMEmbeddings` you need to initialize them by passing a list of token embeddings to it:

```python
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentLSTMEmbeddings([glove_embedding])
```

Now, create an example sentence and call the embedding's `embed()` method.

```python
# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())
```

This will output a single embedding for the complete sentence. The embedding dimensionality depends on the number of
hidden states you are using and whether the LSTM is bidirectional or not.

Note that while `DocumentPoolEmbeddings` are immediately meaningful, `DocumentLSTMEmbeddings` need to be tuned on the
downstream task. This happens automatically in Flair if you train a new model with these embeddings.
`DocumentLSTMEmbeddings` have a number of hyper-parameters that can be tuned to improve learning:

```text
:param hidden_size: the number of hidden states in the lstm.
:param rnn_layers: the number of layers for the lstm.
:param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
layer before putting them into the lstm or not.
:param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
dimension as before will be taken.
:param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not.
:param dropout: the dropout value to be used.
:param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used.
:param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used.
```

## Next 

You can now either look into the tutorial about [loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which
is a pre-requirement for [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
or into [training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).
