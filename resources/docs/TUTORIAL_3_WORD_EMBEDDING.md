# Tutorial 3: Word Embeddings

We provide a set of classes with which you can embed the words in sentences in various ways. This tutorial explains
how that works. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this 
library.


## Embeddings

All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which you need to 
call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains 
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text. All embeddings produced with our methods are PyTorch vectors, so they can be immediately used for training and
fine-tuning.

This tutorial introduces some common embeddings and shows you how to use them. For more details on these embeddings and an overview of all supported embeddings, check [here](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md). 


## Classic Word Embeddings

Classic word embeddings are static and word-level, meaning that each distinct word gets exactly one pre-computed 
embedding. Most embeddings fall under this class, including the popular GloVe or Komninos embeddings. 

Simply instantiate the `WordEmbeddings` class and pass a string identifier of the embedding you wish to load. So, if 
you want to use GloVe embeddings, pass the string 'glove' to the constructor: 

```python
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

# init embedding
glove_embedding = WordEmbeddings('glove')
```
Now, create an example sentence and call the embedding's `embed()` method. You can also pass a list of sentences to
this method since some embedding types make use of batching to increase speed.

```python
# create sentence.
sentence = Sentence('The grass is green .')

# embed a sentence using glove.
glove_embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

This prints out the tokens and their embeddings. GloVe embeddings are PyTorch vectors of dimensionality 100.

You choose which pre-trained embeddings you load by passing the appropriate 
id string to the constructor of the `WordEmbeddings` class. Typically, you use
the **two-letter language code** to init an embedding, so 'en' for English and
'de' for German and so on. By default, this will initialize FastText embeddings trained over Wikipedia.
You can also always use FastText embeddings over Web crawls, by instantiating with '-crawl'. So 'de-crawl' 
to use embeddings trained over German web crawls:

```python
german_embedding = WordEmbeddings('de-crawl')
```

Check out the full list of all word embeddings models [here](/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md), along with more explanations on this class.

We generally recommend the FastText embeddings, or GloVe if you want a smaller model.


## Flair Embeddings

Contextual string embeddings are [powerful embeddings](https://www.aclweb.org/anthology/C18-1139/)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and
thus fundamentally model words as sequences of characters. And (2) they are **contextualized** by their
surrounding text, meaning that the *same word will have different embeddings depending on its
contextual use*.

With Flair, you can use these embeddings simply by instantiating the appropriate embedding class, same as standard word embeddings:

```python
from flair.embeddings import FlairEmbeddings

# init embedding
flair_embedding_forward = FlairEmbeddings('news-forward')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
flair_embedding_forward.embed(sentence)
```

You choose which embeddings you load by passing the appropriate string to the constructor of the `FlairEmbeddings` class. For all supported languages, there is a forward and a backward model. You can load a model for a language by using the **two-letter language code** followed by a hyphen and either **forward** or **backward**. So, if you want to load the forward and backward Flair models for German, do it like this: 

```python
# init forward embedding for German
flair_embedding_forward = FlairEmbeddings('de-forward')
flair_embedding_backward = FlairEmbeddings('de-backward')
```

Check out the full list of all pre-trained FlairEmbeddings models [here](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md), along with more information on standard usage.

## Stacked Embeddings

Stacked embeddings are one of the most important concepts of this library. You can use them to combine different
embeddings together, for instance if you want to use both traditional embeddings together with contextual string
embeddings. Stacked embeddings allow you to mix and match. We find that a combination of embeddings often gives best results. 

All you need to do is use the `StackedEmbeddings` class and instantiate it by passing a list of embeddings that you wish 
to combine. For instance, lets combine classic GloVe embeddings with forward and backward Flair embeddings. This is a combination that we generally recommend to most users, especially for sequence labeling.

First, instantiate the two embeddings you wish to combine: 

```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# init standard GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init Flair forward and backwards embeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these two embeddings.

```python
from flair.embeddings import StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                       ])
```


That's it! Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
sentence = Sentence('The grass is green .')

# just embed a sentence using the StackedEmbedding as you would with any single embedding.
stacked_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

Words are now embedded using a concatenation of three different embeddings. This means that the resulting embedding
vector is still a single PyTorch vector.

## Next 

To get more details on these embeddings and a full overview of all word embeddings that we support, you can look into this 
[tutorial](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md). You can also skip details on word embeddings and go directly to [document embeddings](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) that let you embed entire text
passages with one vector for tasks such as text classification. You can also go directly to the tutorial about
[loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which is a pre-requirement for
[training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).
