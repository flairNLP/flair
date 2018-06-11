

![alt text](resources/docs/flair_logo.svg):

a very simple framework for **state-of-the-art NLP**. Developed by [Zalando Research](https://research.zalando.com/).

---

The framework's strength resides in **hyper-powerful word embeddings** that, when used in 
vanilla classifiaction approaches, yield state-of-the-art NLP components. 

Use `Flair` if:

* you want to reproduce our experiments in paper [(Akbik et. al, 2018)](https://drive.google.com/file/d/19dWlBaoDGXeiZ2wQzxd95-i2nJZnTGWU/view)
* you want to build your own state-of-the-art NLP components based on **contextual string embeddings**, or stacked combinations of various embedding types
* you want to apply our pre-trained models for NER, PoS tagging and chunking to your text


## Flair NLP

`Flair` outperforms the previous best methods on a range of NLP tasks:

| Task | Dataset | Our Result | Previous best |
| -------------    | ------------- | ------------- | ------------- |
| Named Entity Recognition (English) | Conll-03    |  **93.09** (F1)  | *92.22* |
| Named Entity Recognition (German)  | Conll-03    |  **86.85** (F1)  | *78.76* |
| Part-of-Speech tagging | WSJ  | **97.82**  | |
| Chunking | Conll-2000  |  **96.72** (F1) | *96.36* |


## Set up

### Python 3.6

This library runs on Python 3.6 - because methods signatures and type hints are beautiful. If you do not have 
Python 3.6, install it first. [Here is how for Ubuntu 16.04](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/). 


### Requirements

The `requirements.txt` lists all libraries that we depend on. Install them first. For instance, if you 
use pip and virtualenv, run these commands: 
```bash
virtualenv --python=[path/to/python3.6] [path/to/this/virtualenv]

source [path/to/this/virtualenv]/bin/activate

pip install -r requirements.txt
```

### Train a quick model and predict

To check if everything works, go into the virtualenv and run the following command to train a small NER model:
```bash
python train.py 
```
Run this for a few epochs. Then test it on a sentence using the following command:

```bash
python predict.py
```


## Understand

Let's look into some core functionality to understand the library better.

### NLP base types

First, you need to construct Sentence objects for your text.

```python
# The sentence objects holds a sentence that we may want to embed
from flair.data import Sentence

# Make a sentence object by passing a whitespace tokenized string
sentence: Sentence = Sentence('The grass is green .')

# Print the object to see what's in there
print(sentence)

# The Sentence object has a list of Token objects (each token represents a word)
for token in sentence.tokens:
    print(token)

# add a tag to a word in the sentence
sentence.get_token(4).add_tag('ner', 'color')

# print the sentence with all tags of this type
print(sentence.to_tag_string('ner'))

```

### Word Embeddings

Now, you can embed the words in a sentence. We start with a simple example that uses GloVe embeddings:

```python

# all embeddings inherit from the TextEmbedding class. Init a simple glove embedding.
from flair.embeddings import TextEmbedding, WordEmbeddingGensim
glove_embedding: TextEmbedding = WordEmbeddingGensim(precomputed_embeddings_file='resources/embeddings/glove.gensim')

# embed a sentence using glove.
from flair.data import Sentence
sentence: Sentence = Sentence('The grass is green .')
glove_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

### Contextual String Embeddings

You can also use our contextual string embeddings. Same code as above, with different TextEmbedding class:

```python

# the CharLMEmbedding also inherits from the TextEmbedding class
from flair.embeddings import TextEmbedding, CharLMEmbedding
contextual_string_embedding: TextEmbedding = CharLMEmbedding(model_file='resources/LMs/news-forward-2048/lm.pt')

# embed a sentence using CharLM.
from flair.data import Sentence
sentence: Sentence = Sentence('The grass is green .')
contextual_string_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```


### Stacked Embeddings

Very often, you want to combine different embedding types. For instance, you might want to combine classic static
word embeddings such as GloVe with embeddings from a forward and backward language model. This normally gives best
results.

For this case, use the StackedEmbeddings class which combines a list of TextEmbeddings.

```python

# the CharLMEmbedding also inherits from the TextEmbedding class
from flair.embeddings import TextEmbedding, WordEmbeddingGensim, CharLMEmbedding

# init GloVe embedding
glove_embedding: TextEmbedding = WordEmbeddingGensim(precomputed_embeddings_file='resources/embeddings/glove.gensim')

# init CharLM embedding
charlm_embedding_forward: TextEmbedding = CharLMEmbedding(model_file='resources/LMs/news-forward-2048/lm.pt')
charlm_embedding_backward: TextEmbedding = CharLMEmbedding(model_file='resources/LMs/news-backward-2048/lm.pt')

# now create the StackedEmbedding object that combines all embeddings
from flair.embeddings import StackedEmbedding
stacked_embeddings: TextEmbedding = StackedEmbedding(embeddings=[glove_embedding, charlm_embedding_forward, charlm_embedding_backward])


# just embed a sentence using the StackedEmbedding as you would with any single embedding.
from flair.data import Sentence
sentence: Sentence = Sentence('The grass is green .')
stacked_embeddings.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

