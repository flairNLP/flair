

![alt text](resources/docs/flair_logo.svg)

A very simple framework for **state-of-the-art NLP**. Developed by [Zalando Research](https://research.zalando.com/).

---

Flair is:

* **A powerful syntactic / semantic tagger.** Flair allows you to apply our state-of-the-art models for named entity recognition (NER), 
part-of-speech tagging (PoS) and chunking to your text.

* **A text embedding library.** Flair has simple interfaces that allow you to use and combine different word embeddings. 
In particular, you can try out our proposed 
**[contextual string embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)** 
to build your own state-of-the-art NLP methods.

* **A Pytorch NLP framework.** Our framework builds directly on [Pytorch](https://pytorch.org/), making it easy to train your own models and 
experiment with new approaches using Flair embeddings and classes.


## Comparison with State-of-the-Art

Flair outperforms the previous best methods on a range of NLP tasks:

| Task | Dataset | Our Result | Previous best |
| -------------------------------    | ----------- | ---------------- | ------------- |
| Named Entity Recognition (English) | Conll-03    |  **93.09** (F1)  | *92.22 [(Peters et al., 2018)](https://arxiv.org/pdf/1802.05365.pdf)* |
| Named Entity Recognition (English) | Ontonotes   |  **89.71** (F1)  | *86.28 [(Chiu et al., 2016)](https://arxiv.org/pdf/1511.08308.pdf)* |
| Named Entity Recognition (German)  | Conll-03    |  **88.32** (F1)  | *78.76 [(Lample et al., 2016)](https://arxiv.org/abs/1603.01360)* |
| Named Entity Recognition (German)  | Germeval    |  **84.65** (F1)  | *79.08 [(Hänig et al, 2014)](http://asv.informatik.uni-leipzig.de/publication/file/300/GermEval2014_ExB.pdf)*|
| Part-of-Speech tagging | WSJ  | **97.85**  | *97.64 [(Choi, 2016)](https://www.aclweb.org/anthology/N16-1031)*|
| Chunking | Conll-2000  |  **96.72** (F1) | *96.36 [(Peters et al., 2017)](https://arxiv.org/pdf/1705.00108.pdf)*

Here's how to [reproduce these numbers](/resources/docs/EXPERIMENTS.md) using Flair. You can also find a detailed evaluation and discussion in our paper: 

*[Contextual String Embeddings for Sequence Labeling](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing).
Alan Akbik, Duncan Blythe and Roland Vollgraf. 
27th International Conference on Computational Linguistics, COLING 2018.* 


## Quick Start

### Requirements and Installation

The project is based on PyTorch 0.4+ and Python 3.6+, because methods signatures and type hints are beautiful.
If you do not have Python 3.6, install it first. [Here is how for Ubuntu 16.04](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/). 

You can create an appropriate environment using virtualenv:

```
virtualenv --python=[path/to/python3.6] [path/to/this/virtualenv]

source [path/to/this/virtualenv]/bin/activate
```

Within this environment, install the package from the repository:

```
pip install git+https://github.com/zalandoresearch/flair
```

### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is make a `Sentence`, load 
a pre-trained model and use it to predict tags for the sentence:

```python
from flair.data import Sentence
from flair.tagging_model import SequenceTagger

# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)
```

Done! The `Sentence` now has entity annotations. Print the sentence to see what the tagger found.

```python
print(sentence)
print('The following NER tags are found:')
print(sentence.to_tagged_string())
```

This should print: 

```console
Sentence: "I love Berlin ." - 4 Tokens

The following NER tags are found: 

I love Berlin <S-LOC> .
```


## Tutorial

We provide a set of quick tutorials to get you started with the library:

* [Tutorial 1: Basics](/resources/docs/TUTORIAL_BASICS.md)
* [Tutorial 2: Tagging your Text](/resources/docs/TUTORIAL_TAGGING.md)
* [Tutorial 3: Using Word Embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md)
* [Tutorial 4: Using Text Embeddings](/resources/docs/TUTORIAL_TEXT_EMBEDDINGS.md)
* [Tutorial 5: Training your own Models](/resources/docs/TUTORIAL_TRAINING_A_MODEL.md)
 
The tutorials explain how the base NLP classes work, how you can load pre-trained models to tag your
text, how you embed your text with different word embeddings, and how you can train your own 
sequence labeling models. Let us know if anything is unclear.


## Citing Flair

Please cite the following paper when using Flair: 

```
@inproceedings{akbik2018coling,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {(forthcoming)},
  year      = {2018}
}
```

## Contact 

Please email your questions or comments to [Alan Akbik](http://alanakbik.github.io/).

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; 
start with our [contributor guidelines](/resources/docs/CONTRIBUTING.md) and then 
check these [open issues](https://github.com/zalandoresearch/flair/issues) for specific tasks.

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit 
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around 
the code should hopefully be easy.


## [License](/LICENSE)

The MIT License (MIT)

Flair is licensed under the following MIT license: The MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

