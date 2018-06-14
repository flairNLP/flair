

![alt text](resources/docs/flair_logo.svg)

- a very simple framework for **state-of-the-art NLP**. Developed by [Zalando Research](https://research.zalando.com/).

---

The framework's strength resides in **hyper-powerful word embeddings** that, when used in 
vanilla classifiaction approaches, yield state-of-the-art NLP components. 

Use `Flair` if:

* you want to easily **embed your text** with various word embeddings, including *contextual string embeddings*, to build your own state-of-the-art NLP components
* you want to reproduce our experiments in paper [Akbik et. al (2018)](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
* you want to apply our pre-trained models for NER, PoS tagging and chunking to your text


## Flair NLP

`Flair` outperforms the previous best methods on a range of NLP tasks:

| Task | Dataset | Our Result | Previous best |
| -------------    | ------------- | ------------- | ------------- |
| Named Entity Recognition (English) | Conll-03    |  **93.09** (F1)  | *92.22 [(Peters et al., 2018)](https://arxiv.org/pdf/1802.05365.pdf)* |
| Named Entity Recognition (English) | Ontonotes    |  **89.71** (F1)  | *86.28 [(Chiu et al., 2016)](https://arxiv.org/pdf/1511.08308.pdf)* |
| Named Entity Recognition (German)  | Conll-03    |  **88.32** (F1)  | *78.76 [(Lample et al., 2016)](https://arxiv.org/abs/1603.01360)* |
| Named Entity Recognition (German)  | Germeval    |  **84.65** (F1)  | *79.08 [(Hänig et al, 2014)](http://asv.informatik.uni-leipzig.de/publication/file/300/GermEval2014_ExB.pdf)*|
| Part-of-Speech tagging | WSJ  | **97.85**  | *97.64 [(Choi, 2016)](https://www.aclweb.org/anthology/N16-1031)*|
| Chunking | Conll-2000  |  **96.72** (F1) | *96.36 [(Peters et al., 2017)](https://arxiv.org/pdf/1705.00108.pdf)*


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

### Training Data

In order to train your own models, you need training data. For instance, get the publicly available 
CoNLL-03 data set for English and place train, test and dev data in `/resources/tasks/conll_03/` as follows: 

```
/resources/tasks/conll_03/eng.testa
/resources/tasks/conll_03/eng.testb
/resources/tasks/conll_03/eng.train
```

To set up more tasks, follow the guidelines here.

### Train a quick model and predict

Once you have the data, go into the virtualenv and run the following command to train a small NER model:
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
sentence = Sentence('The grass is green .')

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

# all embeddings inherit from the TextEmbeddings class. Init a simple glove embedding.
from flair.embeddings import WordEmbeddings
glove_embedding = WordEmbeddings('glove')

# embed a sentence using glove.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
glove_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

### Contextual String Embeddings

You can also use our contextual string embeddings. Same code as above, with different TextEmbedding class:

```python

# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import CharLMEmbeddings
contextual_string_embedding = CharLMEmbeddings('news-forward')

# embed a sentence using CharLM.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
contextual_string_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

This is just one example, the [tutorial](/resources/docs/TUTORIAL.md) contains more!

## [Tutorial](/resources/docs/TUTORIAL.md)

Flair makes it easy to embed your text with many different types of embeddings and their combinations. And then train 
state-of-the-art NLP models. Just follow the [tutorial](/resources/docs/TUTORIAL.md) to get a better overview of functionality.

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; 
start with our [contributor guidelines](/resources/docs/CONTRIBUTING.md) and then 
check these [open issues](https://github.com/zalandoresearch/flair/issues) for specific tasks.

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit 
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around 
the code should hopefully be easy.


## [License](/LICENSE.md)

Flair is in general licensed under the following MIT license: The MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
