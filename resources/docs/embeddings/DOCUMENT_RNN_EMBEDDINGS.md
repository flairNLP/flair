## Document RNN Embeddings

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

### RNN type 

If you want to use a different type of RNN, you need to set the `rnn_type` parameter in the constructor. So,
to initialize a document RNN embedding with an LSTM, do:

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_lstm_embeddings = DocumentRNNEmbeddings([glove_embedding], rnn_type='LSTM')
```

### Need to be trained on a task

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
