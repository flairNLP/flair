# Tutorial 5: Document Embeddings

Document embeddings are different from [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) in that they
give you one embedding for an entire text, whereas word embeddings give you embeddings for individual words.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work.

## Embeddings

All document embedding classes inherit from the `DocumentEmbeddings` class and implement the `embed()` method which you
need to call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are PyTorch vectors, so they can be immediately used for training and
fine-tuning. There are three main document embeddings in Flair: (1) DocumentPoolEmbeddings that simply do an average over all word embeddings in the sentence, (2) DocumentRNNEmbeddings that train an RNN over all word embeddings in a sentence, and (3) TransformerDocumentEmbeddings that use a pre-trained transformer: 

```python
# document embedding is a mean over GloVe word embeddings
pooled_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove')], pooling='mean')

# document embedding is an LSTM over GloVe word embeddings
lstm_embeddings = DocumentRNNEmbeddings([WordEmbeddings('glove')], rnn_type='lstm')

# document embedding is a pre-trained transformer
transformer_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
```

Simply call embed() to embed your sentence with one of these three options: 

```python
# example sentence
sentence = Sentence('The grass is green.')

# embed with pooled embeddings
pooled_embeddings.embed(sentence)

# print embedding for whole sentence
print(sentence.embedding)
```

## DocumentPoolEmbeddings

DocumentPoolEmbeddings calculate a pooling operation over all word embeddings in a document.
The default operation is `mean` which gives us the mean of all words in the sentence.
The resulting embedding is taken as document embedding.

To create a mean document embedding simply create any number of `TokenEmbeddings` first and put them in a list.
Afterwards, initiate the `DocumentPoolEmbeddings` with this list of `TokenEmbeddings`.
So, if you want to create a document embedding using GloVe embeddings together with `FlairEmbeddings`,
use the following code:

```python
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding])
```

Now, create an example sentence and call the embedding's `embed()` method.

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
document_embeddings = DocumentPoolEmbeddings([glove_embedding],
                                             pooling='min')
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

## DocumentRNNEmbeddings

Besides simple pooling we also support a method based on an RNN to obtain a `DocumentEmbeddings`.
The RNN takes the word embeddings of every token in the document as input and provides its last output state as document
embedding. You can choose which type of RNN you wish to use.

In order to use the `DocumentRNNEmbeddings` you need to initialize them by passing a list of token embeddings to it:

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentRNNEmbeddings([glove_embedding])
```

By default, a GRU-type RNN is instantiated. Now, create an example sentence and call the embedding's `embed()` method.

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

If you want to use a different type of RNN, you need to set the `rnn_type` parameter in the constructor. So,
to initialize a document RNN embedding with an LSTM, do:

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_lstm_embeddings = DocumentRNNEmbeddings([glove_embedding], rnn_type='LSTM')
```

Note that while `DocumentPoolEmbeddings` are immediately meaningful, `DocumentRNNEmbeddings` need to be tuned on the
downstream task. This happens automatically in Flair if you train a new model with these embeddings. You can find an example of training a text classification model [here](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md#training-a-text-classification-model). Once the model is trained, you can access the tuned `DocumentRNNEmbeddings` object directly from the classifier object and use it to embed sentences.

```python
document_embeddings = classifier.document_embeddings

sentence = Sentence('The grass is green . And the sky is blue .')

document_embeddings.embed(sentence)

print(sentence.get_embedding())
```

`DocumentRNNEmbeddings` have a number of hyper-parameters that can be tuned to improve learning:

```text
:param hidden_size: the number of hidden states in the rnn.
:param rnn_layers: the number of layers for the rnn.
:param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
layer before putting them into the rnn or not.
:param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
dimension as before will be taken.
:param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not.
:param dropout: the dropout value to be used.
:param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used.
:param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used.
:param rnn_type: one of 'RNN' or 'LSTM'
```


## TransformerDocumentEmbeddings

You can get embeddings for a whole sentence directly from a pre-trained [`transformer`](https://github.com/huggingface/transformers). There is a single class for all transformer embeddings that you instantiate with different identifiers get different transformers. For instance, to load a standard BERT transformer model, do:  

```python
from flair.embeddings import TransformerDocumentEmbeddings

# init embedding
embedding = TransformerDocumentEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed the sentence
embedding.embed(sentence)
```

If instead you want to use RoBERTa, do: 

```python
from flair.embeddings import TransformerDocumentEmbeddings

# init embedding
embedding = TransformerDocumentEmbeddings('roberta-base')

# create a sentence
sentence = Sentence('The grass is green .')

# embed the sentence
embedding.embed(sentence)
```

[Here](https://huggingface.co/transformers/pretrained_models.html) is a full list of all models (BERT, RoBERTa, XLM, XLNet etc.). You can use any of these models with this class. 


## Next

You can now either look into the tutorial about [loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which
is a pre-requirement for [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
or into [training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).
