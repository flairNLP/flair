# Tutorial 4: Document Embeddings

Document embeddings are different from [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) in that they 
give you one embedding for an entire text, whereas word embeddings give you embeddings for individual words. 

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) work.

# Embeddings

All document embedding classes inherit from the `DocumentEmbeddings` class and implement the `embed()` method which you need to call 
to embed your text. This means that for most users of Flair, the complexity of different embeddings remains hidden 
behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are pytorch vectors, so they can be immediately used for training and 
fine-tuning.

# Document Embeddings

Document embeddings define one embedding for an entire document.
Every document embedding takes any number of word embedding as input.
The word embeddings are than mapped to a single text embedding.
Currently, we have two different methods defined on how to obtain the document embedding from a list of word embeddings.

### MEAN

The first method calculates the mean over all word embeddings in a document.
The resulting embedding is taken as document embedding.

To create a mean document embedding simply create any number of `TokenEmbeddings` first.
Afterwards, initiate the `DocumentMeanEmbeddings` and pass a list containing the created `WordEmbeddings`.
So, if you want to create a document embedding using GloVe embeddings together with CharLMEmbeddings,
use the following code:

```python
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentMeanEmbeddings

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')

# initialize the text embeddings
document_embeddings = DocumentMeanEmbeddings([glove_embedding, charlm_embedding_backward, charlm_embedding_forward])
```

Now, create an example sentence and call the embedding's `embed()` method. 
You always pass a list of sentences to this method since some embedding types make use of batching to increase speed. 
So if you only have one sentence, pass a list containing only one sentence:

```python
from flair.data import Sentence

# create an example sentence and embed it
sentence = Sentence('The grass is green .')
document_embeddings.embed(sentences=[sentence])

# now check out the embedded tokens.
print(sentence.get_embedding())
```

This prints out the embedding of the text. 
The embeddings dimensionality depends on the dimensionality of word embeddings you are using.

### LSTM

The second method creates a `DocumentEmbeddings` by using a LSTM.
The method calculates first word embeddings for every token in the document.
Those word embeddings are then taken as input to a LSTM.
In the end, the last representation of the LSTM is taken as the document embedding.

To create a `DocumentLSTMEmbeddings` simply create any number of `TokenEmbeddings` first.
Afterwards, initiate the `DocumentLSTMEmbeddings` and pass a list containing the created WordEmbeddings.
If you want, you can also specify some other parameters:
```text
:param hidden_states: the number of hidden states in the lstm
:param num_layers: the number of layers for the lstm
:param reproject_words: boolean value, indicating whether to reproject the word embedding in a separate linear
layer before putting them into the lstm or not
:param reproject_words_dimension: output dimension of reprojecting words
:param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not
:param use_first_representation: boolean value, indicating whether to concatenate the first and last
representation of the lstm to be used as final document embedding or not
```

So if you want to create a text embedding using only GloVe embeddings, use the following code:

```python
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentLSTMEmbeddings([glove_embedding])
```

Now, create an example sentence and call the embedding's `embed()` method. 
You always pass a list of sentences to this method since some embedding types make use of batching to increase speed. 
So if you only have one sentence, pass a list containing only one sentence:

```python
from flair.data import Sentence

sentence = Sentence('The grass is green .')
document_embeddings.embed(sentences=[sentence])

# now check out the embedded tokens.
print(sentence.get_embedding())
```

This prints out the embedding of the text. 
The embedding dimensionality depends on the number of hidden states you are using and whether the LSTM is bidirectional or not.

## Next 

You can now either look into the tutorial about [training your own models](/resources/docs/TUTORIAL_TRAINING_A_MODEL.md). 