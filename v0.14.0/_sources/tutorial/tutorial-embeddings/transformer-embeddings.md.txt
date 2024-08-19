# Transformer embeddings

Flair supports various Transformer-based architectures like BERT or XLNet from [HuggingFace](https://github.com/huggingface), 
with two classes [`TransformerWordEmbeddings`](#flair.embeddings.token.TransformerWordEmbeddings) (to embed words) and [`TransformerDocumentEmbeddings`](#flair.embeddings.document.TransformerDocumentEmbeddings) (to embed documents).

## Embeddings words 

For instance, to load a standard BERT transformer model, do:

```python
from flair.embeddings import TransformerWordEmbeddings

# init embedding
embedding = TransformerWordEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

If instead you want to use RoBERTa, do:

```python
from flair.embeddings import TransformerWordEmbeddings

# init embedding
embedding = TransformerWordEmbeddings('roberta-base')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

Use the [Huggingface Model hub](https://huggingface.co/models) to find any open source text embedding model to use.


## Embeddings sentences

To embed a whole sentence as one (instead of each word in the sentence), simply use the [`TransformerDocumentEmbeddings`](#flair.embeddings.document.TransformerDocumentEmbeddings) 
instead:

```python
from flair.embeddings import TransformerDocumentEmbeddings

# init embedding
embedding = TransformerDocumentEmbeddings('roberta-base')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

## Arguments

There are several options that you can set when you init the [`TransformerWordEmbeddings`](#flair.embeddings.token.TransformerWordEmbeddings) 
and [`TransformerDocumentEmbeddings`](#flair.embeddings.document.TransformerDocumentEmbeddings) classes:

| Argument               | Default             | Description                                                                                                                                
|------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------
| `model`                | `bert-base-uncased` | The string identifier of the transformer model you want to use (see above)                                                                 |
| `layers`               | `all`               | Defines the layers of the Transformer-based model that produce the embedding                                                               |
| `subtoken_pooling`     | `first`             | See [Pooling operation section](#pooling).                                                                                                 |
| `layer_mean`           | `True`              | See [Layer mean section](#layer-mean).                                                                                                     |
| `fine_tune`            | `False`             | Whether or not embeddings are fine-tuneable.                                                                                               |
| `allow_long_sentences` | `True`              | Whether or not texts longer than maximal sequence length are supported.                                                                    |
| `use_context`          | `False`             | Set to True to include context outside of sentences. This can greatly increase accuracy on some tasks, but slows down embedding generation |


### Layers

The `layers` argument controls which transformer layers are used for the embedding. If you set this value to '-1,-2,-3,-4', the top 4 layers are used to make an embedding. If you set it to '-1', only the last layer is used. If you set it to "all", then all layers are used.

This affects the length of an embedding, since layers are just concatenated.

```python
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

sentence = Sentence('The grass is green.')

# use only last layers
embeddings = TransformerWordEmbeddings('bert-base-uncased', layers='-1', layer_mean=False)
embeddings.embed(sentence)
print(sentence[0].embedding.size())

sentence.clear_embeddings()

# use last two layers
embeddings = TransformerWordEmbeddings('bert-base-uncased', layers='-1,-2', layer_mean=False)
embeddings.embed(sentence)
print(sentence[0].embedding.size())

sentence.clear_embeddings()

# use ALL layers
embeddings = TransformerWordEmbeddings('bert-base-uncased', layers='all', layer_mean=False)
embeddings.embed(sentence)
print(sentence[0].embedding.size())
```

This should print:
```console
torch.Size([768])
torch.Size([1536])
torch.Size([9984])
```

I.e. the size of the embedding increases the more layers we use (but ONLY if layer_mean is set to False, otherwise the length is always the same).

(pooling)=
### Pooling operation

Most of the Transformer-based models (except Transformer-XL) use subword tokenization. E.g. the following
token `puppeteer` could be tokenized into the subwords: `pupp`, `##ete` and `##er`.

We implement different pooling operations for these subwords to generate the final token representation:

* `first`: only the embedding of the first subword is used
* `last`: only the embedding of the last subword is used
* `first_last`: embeddings of the first and last subwords are concatenated and used
* `mean`: a `torch.mean` over all subword embeddings is calculated and used

You can choose which one to use by passing this in the constructor:

```python
# use first and last subtoken for each word
embeddings = TransformerWordEmbeddings('bert-base-uncased', subtoken_pooling='first_last')
embeddings.embed(sentence)
print(sentence[0].embedding.size())
```

(layer-mean)=
### Layer mean

The Transformer-based models have a certain number of layers. By default, all layers you select are
concatenated as explained above. Alternatively, you can set layer_mean=True to do a mean over all
selected layers. The resulting vector will then always have the same dimensionality as a single layer:

```python
from flair.embeddings import TransformerWordEmbeddings

# init embedding
embedding = TransformerWordEmbeddings("roberta-base", layers="all", layer_mean=True)

# create a sentence
sentence = Sentence("The Oktoberfest is the world's largest Volksfest .")

# embed words in sentence
embedding.embed(sentence)
```

### Fine-tuneable or not

In some setups, you may wish to fine-tune the transformer embeddings. In this case, set `fine_tune=True` in the init method.
When fine-tuning, you should also only use the topmost layer, so best set `layers='-1'`.

```python
# use first and last subtoken for each word
embeddings = TransformerWordEmbeddings('bert-base-uncased', fine_tune=True, layers='-1')
embeddings.embed(sentence)
print(sentence[0].embedding)
```

This will print a tensor that now has a gradient function and can be fine-tuned if you use it in a training routine.

```python
tensor([-0.0323, -0.3904, -1.1946,  ...,  0.1305, -0.1365, -0.4323],
       device='cuda:0', grad_fn=<CatBackward>)
```

### Models

Please have a look at the awesome [Huggingface Model hub](https://huggingface.co/models) to find any open source text embedding model to use.

