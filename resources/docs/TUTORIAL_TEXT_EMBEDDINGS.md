# Tutorial 4: Text Embeddings

Text embeddings are different from [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) in that they give you one embedding for an entire text, whereas word embeddings give you embeddings for individual words. 

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this library and how
   [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) work.


# Embeddings

All embedding classes inherit from the `TextEmbeddings` class and implement the `embed()` method which you need to call 
to embed your text. This means that for most users of Flair, the complexity of different embeddings remains hidden 
behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are pytorch vectors, so they can be immediately used for training and 
fine-tuning.

# Text Embeddings

Text embeddings define one embedding for an entire text.
Every text embedding take any number of word embedding as input.
The word embeddings are than mapped to a single text embedding.
Currently, we have two different methods defined on how to obtain the text embedding from a list of word embeddings.

### MEAN

The first method calculates the mean over all word embeddings in a text.
The resulting embedding is taken as text embedding.

To create a mean text embedding simply create any number of WordEmbeddings first.
Afterwards, initiate the TextMeanEmbedder and pass a list containing the created WordEmbeddings.
So if you want to create a text embedding using GloVe embeddings together with CharLMEmbeddings,
use the following code:

```python
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, TextMeanEmbedder

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')

# initialize the text embeddings
text_embeddings = TextMeanEmbedder([glove_embedding, charlm_embedding_backward, charlm_embedding_forward])
```

Now, create an example sentence and call the embedding's `embed()` method. 
You always pass a list of sentences to this method since some embedding types make use of batching to increase speed. 
So if you only have one sentence, pass a list containing only one sentence:

```python
from flair.data import Sentence

# create an example sentence and embed it
sentence = Sentence('The grass is green .')
text_embeddings.embed(paragraphs=[sentence])

# now check out the embedded tokens.
print(sentence.get_embedding())
```

This prints out the embedding of the text. 
The embeddings dimensionality depends on the dimensionality of word embeddings you are using.

### LSTM

The second method obtains a text embeddings by using an LSTM.
The method calculates first word embeddings for every token in the text.
Those word embeddings are then taken as input to the LSTM.
In the end, the last representation of the LSTM is taken as the text embedding.

To create a LSTM text embedding simply create any number of WordEmbeddings first.
Afterwards, initiate the TextLSTMEmbedder and pass a list containing the created WordEmbeddings.
If you want, you can also specify some other parameters:
```bash
:param hidden_states: the number of hidden states in the lstm
:param num_layers: the number of layers for the lstm
:param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not
:param reproject_words: boolean value, indicating whether to reproject the word embedding in a separate linear layer before putting them into the lstm or not
```

So if you want to create a text embedding using only GloVe embeddings, use the following code:

```python
from flair.embeddings import WordEmbeddings, TextLSTMEmbedder

glove_embedding = WordEmbeddings('glove')

text_embeddings = TextLSTMEmbedder([glove_embedding])
```

Now, create an example sentence and call the embedding's `embed()` method. 
You always pass a list of sentences to this method since some embedding types make use of batching to increase speed. 
So if you only have one sentence, pass a list containing only one sentence:

```python
from flair.data import Sentence

sentence = Sentence('The grass is green .')
text_embeddings.embed(paragraphs=[sentence])

# now check out the embedded tokens.
print(sentence.get_embedding())
```

This prints out the embedding of the text. 
The embedding dimensionality depends on the number of hidden states you are using and whether the LSTM is bidirectional or not.

## Next 

You can now either look into the tutorial about [training your own models](/resources/docs/TUTORIAL_TRAINING_A_MODEL.md). 