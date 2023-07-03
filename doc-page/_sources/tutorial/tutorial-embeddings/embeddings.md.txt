# Embeddings

This tutorial shows you how to use Flair to produce **embeddings** for words and documents. Embeddings
are vector representations that are useful for a variety of reasons. All Flair models are trained on 
top of embeddings, so if you want to train your own models, you should understand how embeddings work.

## Example 1: Embeddings Words with Transformers

Let's use a standard BERT model (bert-base-uncased) to embed the sentence "the grass is green".

Simply instantiate [`TransformerWordEmbeddings`](#flair.embeddings.token.TransformerWordEmbeddings) and call [`embed()`](#flair.embeddings.base.Embeddings.embed) over an example sentence: 

```python
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

# init embedding
embedding = TransformerWordEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

This will cause **each word in the sentence** to be embedded. You can iterate through the words and get 
each embedding like this:

```python
# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

This will print each token as a long PyTorch vector: 
```console
Token[0]: "The"
tensor([-0.0323, -0.3904, -1.1946,  0.1296,  0.5806, ..], device='cuda:0')
Token[1]: "grass"
tensor([-0.3973,  0.2652, -0.1337,  0.4473,  1.1641, ..], device='cuda:0')
Token[2]: "is"
tensor([ 0.1374, -0.3688, -0.8292, -0.4068,  0.7717, ..], device='cuda:0')
Token[3]: "green"
tensor([-0.7722, -0.1152,  0.3661,  0.3570,  0.6573, ..], device='cuda:0')
Token[4]: "."
tensor([ 0.1441, -0.1772, -0.5911,  0.2236, -0.0497, ..], device='cuda:0')
```

*(Output truncated for readability, actually the vectors are much longer.)*

Transformer word embeddings are the most important concept in Flair. Check out more info in this dedicated chapter.

## Example 2: Embeddings Documents with Transformers

Sometimes you want to have an **embedding for a whole document**, not only individual words. In this case, use one of the 
DocumentEmbeddings classes in Flair. 

Let's again use a standard BERT model to get an embedding for the entire sentence "the grass is green":  

```python
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

# init embedding
embedding = TransformerDocumentEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

Now, the whole sentence is embedded. Print the embedding like this: 

```python
# now check out the embedded sentence
print(sentence.embedding)
```

[`TransformerDocumentEmbeddings`](#flair.embeddings.document.TransformerDocumentEmbeddings) are the most important concept in Flair. Check out more info in [this](project:transformer-embeddings.md) dedicated chapter.


## How to Stack Embeddings

Flair allows you to combine embeddings into "embedding stacks". When not fine-tuning, using combinations of embeddings often gives best results!

Use the [`StackedEmbeddings`](#flair.embeddings.token.StackedEmbeddings) class and instantiate it by passing a list of embeddings that you wish to combine. For instance, lets combine classic GloVe [`WordEmbeddings`](#flair.embeddings.token.WordEmbeddings) with forward and backward [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings). 

First, instantiate the two embeddings you wish to combine:

```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# init standard GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init Flair forward and backwards embeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
```

Now instantiate the [`StackedEmbeddings`](#flair.embeddings.token.StackedEmbeddings) class and pass it a list containing these two embeddings.

```python
from flair.embeddings import StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                       ])
```


That's it! Now just use this embedding like all the other embeddings, i.e. call the [`embed()`](#flair.embeddings.base.Embeddings.embed) method over your sentences.

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




