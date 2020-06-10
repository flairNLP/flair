# Document Pool Embeddings

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
document_embeddings = DocumentPoolEmbeddings([glove_embedding],  pooling='min')
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
