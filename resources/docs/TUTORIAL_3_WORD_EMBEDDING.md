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

Simply instantiate the WordEmbeddings class and pass a string identifier of the embedding you wish to load. So, if 
you want to use GloVe embeddings, pass the string 'glove' to the constructor: 

```python
from flair.embeddings import WordEmbeddings

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

This prints out the tokens and their embeddings. GloVe embeddings are Pytorch vectors of dimensionality 100.

You choose which pre-trained embeddings you load by passing the appropriate 
id string to the constructor of the `WordEmbeddings` class. Typically, you use
the **two-letter language code** to init an embedding, so 'en' for English and
'de' for German and so on. By default, this will initialize FastText embeddings trained over Wikipedia.
You can also always use FastText embeddings over Web crawls, by instantiating with '-crawl'. So 'de-crawl' 
to use embeddings trained over German web crawls:

```python
german_embedding = WordEmbeddings('de-crawl')
```

We generally recommend the FastText embeddings, or GloVe if you want a smaller model.


## Flair Embeddings


## Stacked Embeddings

Stacked embeddings are one of the most important concepts of this library. You can use them to combine different
embeddings together, for instance if you want to use both traditional embeddings together with contextual string
embeddings (see next chapter).
Stacked embeddings allow you to mix and match. We find that a combination of embeddings often gives best results. 

All you need to do is use the `StackedEmbeddings` class and instantiate it by passing a list of embeddings that you wish 
to combine. For instance, lets combine classic GloVe embeddings with character embeddings. This is effectively the architecture proposed in (Lample et al., 2016).

First, instantiate the two embeddings you wish to combine: 

```python
from flair.embeddings import WordEmbeddings, CharacterEmbeddings

# init standard GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init standard character embeddings
character_embeddings = CharacterEmbeddings()
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these two embeddings.

```python
from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[glove_embedding, character_embeddings])
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

Words are now embedded using a concatenation of two different embeddings. This means that the resulting embedding
vector is still a single PyTorch vector.

## Next 

You can now either look into [BERT, ELMo, and Flair embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md),
or go directly to the tutorial about [loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which is a
pre-requirement for [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).
