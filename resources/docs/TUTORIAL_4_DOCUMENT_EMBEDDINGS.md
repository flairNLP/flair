# Tutorial 4: Document Embeddings

Document embeddings are different from [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) in that they 
give you one embedding for an entire text, whereas word embeddings give you embeddings for individual words. 

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work.

# Embeddings

All document embedding classes inherit from the `DocumentEmbeddings` class and implement the `embed()` method which you need to call 
to embed your text. This means that for most users of Flair, the complexity of different embeddings remains hidden 
behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are Pytorch vectors, so they can be immediately used for training and
fine-tuning.

# Document Embeddings

Our document embeddings are created from the embeddings of all words in the document.
Currently, we have two different methods to obtain a document embedding from a list of word embeddings.

### Pooling

The first method calculates a pooling operation over all word embeddings in a document.
The default operation is 'mean' which gives us the mean of all words in the sentence. 
The resulting embedding is taken as document embedding.

To create a mean document embedding simply create any number of `TokenEmbeddings` first and put them in a list.
Afterwards, initiate the `DocumentMeanEmbeddings` with this list of `TokenEmbeddings`.
So, if you want to create a document embedding using GloVe embeddings together with CharLMEmbeddings,
use the following code:

```python
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentPoolEmbeddings

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
sentence = Sentence('The grass is green .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())
```

This prints out the embedding of the document.
Since the document embedding is derived from word embeddings, its dimensionality depends on the dimensionality of word embeddings you are using.


### LSTM

The second method creates a `DocumentEmbeddings` using an LSTM.
The LSTM takes as input the word embeddings of every token in the document and provides its last output state as document embedding.

Initiate the `DocumentLSTMEmbeddings` by passing a list of word embeddings:

```python
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentLSTMEmbeddings([glove_embedding])
```

Now, create an example sentence and call the embedding's `embed()` method.

```python
# create an example sentence
sentence = Sentence('The grass is green .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())
```

The embedding dimensionality depends on the number of hidden states you are using and whether the LSTM is bidirectional or not.

Note that while MEAN embeddings are immediately meaningful, LSTM embeddings need to be tuned on the downstream task.
This happens automatically in Flair if you train a new model with these embeddings. There are a number of hyperparameters of
the LSTM you can tune to improve learning:

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


## Next 

You can now either look into the tutorial about [training your own models](/resources/docs/TUTORIAL_5_TRAINING_A_MODEL.md) or into [training your own embeddings](/resources/docs/TUTORIAL_6_TRAINING_LM_EMBEDDINGS.md).