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
fine-tuning. There are four main document embeddings in Flair: (1) `DocumentPoolEmbeddings` that simply do an average over all word embeddings in the sentence, (2) `DocumentRNNEmbeddings` that train an RNN over all word embeddings in a sentence, and (3) `TransformerDocumentEmbeddings` / (4) `SentenceTransformerDocumentEmbeddings` that both use a pre-trained transformer: 

```python
from flair.embeddings import TransformerDocumentEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings, SentenceTransformerDocumentEmbeddings

# document embedding is a mean over GloVe word embeddings
pooled_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove')], pooling='mean')

# document embedding is an LSTM over GloVe word embeddings
lstm_embeddings = DocumentRNNEmbeddings([WordEmbeddings('glove')], rnn_type='lstm')

# document embedding is a pre-trained transformer
transformer_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')

# document embedding is a pre-trained transformer
sent_transformer_embeddings = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')
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

We give details on all four document embeddings in the following:

## Document Pool Embeddings

The simplest type of document embedding does a pooling operation over all word embeddings in a sentence to 
obtain an embedding for the whole sentence. The default is mean pooling, meaning that the average of all 
word embeddings is used. 

To instantiate, you need to pass a list of word embeddings to pool over: 

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

This prints out the embedding of the document. Since the document embedding is derived from word embeddings, its dimensionality depends on the dimensionality of word embeddings you are using. For more details on these embeddings, check [here](/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md). 

One advantage of DocumentPoolEmbeddings is that they do not need to be trained, you can immediately use them to embed your documents. 

## Document RNN Embeddings

These embeddings run an RNN over all words in sentence and use the final state of the RNN as embedding for the whole document. In order to use the `DocumentRNNEmbeddings` you need to initialize them by passing a list of token embeddings to it:

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
hidden states you are using and whether the RNN is bidirectional or not. For more details on these embeddings, check [here](/resources/docs/embeddings/DOCUMENT_RNN_EMBEDDINGS.md). 

Note that when you initialize this embedding, the RNN weights are randomly initialized. So this embedding needs to be trained in order to make sense. 

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

## SentenceTransformerDocumentEmbeddings

You can also get several embeddings from the [`sentence-transformer`](https://github.com/UKPLab/sentence-transformers) library. The procedure is similar to the TransformerDocumentEmbeddings class:  

```python
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

# init embedding
embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

# create a sentence
sentence = Sentence('The grass is green .')

# embed the sentence
embedding.embed(sentence)
```

You can find a full list of their pretained models [here](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0).

## Next

You can now either look into the tutorial about [loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which
is a pre-requirement for [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
or into [training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).
