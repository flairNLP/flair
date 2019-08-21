![alt text](resources/docs/flair_logo.svg)

[![PyPI version](https://badge.fury.io/py/flair.svg)](https://badge.fury.io/py/flair)
[![GitHub Issues](https://img.shields.io/github/issues/zalandoresearch/flair.svg)](https://github.com/zalandoresearch/flair/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Travis](https://img.shields.io/travis/zalandoresearch/flair.svg)](https://travis-ci.org/zalandoresearch/flair)

A very simple framework for **state-of-the-art NLP**. Developed by [Zalando Research](https://research.zalando.com/).

---

Flair is:

* **A powerful NLP library.** Flair allows you to apply our state-of-the-art natural language processing (NLP)
models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS),
 sense disambiguation and classification.

* **Multilingual.** Thanks to the Flair community, we support a rapidly growing number of languages. We also now include
'*one model, many languages*' taggers, i.e. single models that predict PoS or NER tags for input text in various languages.

* **A text embedding library.** Flair has simple interfaces that allow you to use and combine different word and 
document embeddings, including our proposed **[Flair embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)**, BERT embeddings and ELMo embeddings.

* **A Pytorch NLP framework.** Our framework builds directly on [Pytorch](https://pytorch.org/), making it easy to 
train your own models and experiment with new approaches using Flair embeddings and classes.

Now at [version 0.4.2](https://github.com/zalandoresearch/flair/releases)!

## Comparison with State-of-the-Art

Flair outperforms the previous best methods on a range of NLP tasks:

| Task | Language | Dataset | Flair | Previous best |
| -------------------------------  | ---  | ----------- | ---------------- | ------------- |
| Named Entity Recognition |English | Conll-03    |  **93.18** (F1)  | *92.22 [(Peters et al., 2018)](https://arxiv.org/pdf/1802.05365.pdf)* |
| Named Entity Recognition |English | Ontonotes   |  **89.3** (F1)  | *86.28 [(Chiu et al., 2016)](https://arxiv.org/pdf/1511.08308.pdf)* |
| Emerging Entity Detection | English | WNUT-17      |  **49.49** (F1)  | *45.55 [(Aguilar et al., 2018)](http://aclweb.org/anthology/N18-1127.pdf)* |
| Part-of-Speech tagging |English| WSJ  | **97.85**  | *97.64 [(Choi, 2016)](https://www.aclweb.org/anthology/N16-1031)*|
| Chunking |English| Conll-2000  |  **96.72** (F1) | *96.36 [(Peters et al., 2017)](https://arxiv.org/pdf/1705.00108.pdf)*
| Named Entity Recognition | German  | Conll-03    |  **88.27** (F1)  | *78.76 [(Lample et al., 2016)](https://arxiv.org/abs/1603.01360)* |
| Named Entity Recognition |German  | Germeval    |  **84.65** (F1)  | *79.08 [(Hänig et al, 2014)](http://asv.informatik.uni-leipzig.de/publication/file/300/GermEval2014_ExB.pdf)*|
| Named Entity Recognition | Dutch  | Conll-03    |  **90.44** (F1)  | *81.74 [(Lample et al., 2016)](https://arxiv.org/abs/1603.01360)* |
| Named Entity Recognition |Polish  | PolEval-2018    |  **86.6** (F1) <br> [(Borchmann et al., 2018)](https://github.com/applicaai/poleval-2018) | *85.1 [(PolDeepNer)](https://github.com/CLARIN-PL/PolDeepNer/)*|

Here's how to [reproduce these numbers](/resources/docs/EXPERIMENTS.md) using Flair. You can also find detailed evaluations and discussions in our papers:

* *[Contextual String Embeddings for Sequence Labeling](https://aclanthology.coli.uni-saarland.de/papers/C18-1139/c18-1139).
Alan Akbik, Duncan Blythe and Roland Vollgraf. 
27th International Conference on Computational Linguistics, **COLING 2018**.*

* *[Pooled Contextualized Embeddings for Named Entity Recognition](https://www.aclweb.org/anthology/papers/N/N19/N19-1078/).
Alan Akbik, Tanja Bergmann and Roland Vollgraf.
2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics, **NAACL 2019**.*

* *[FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP](https://www.aclweb.org/anthology/papers/N/N19/N19-4010/).
Alan Akbik, Tanja Bergmann, Duncan Blythe, Kashif Rasul, Stefan Schweter and Roland Vollgraf.
2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), **NAACL 2019**.*

## Quick Start

### Requirements and Installation

The project is based on PyTorch 0.4+ and Python 3.6+, because method signatures and type hints are beautiful.
If you do not have Python 3.6, install it first. [Here is how for Ubuntu 16.04](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
Then, in your favorite virtual environment, simply do:

```
pip install flair
```

### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is make a `Sentence`, load 
a pre-trained model and use it to predict tags for the sentence:

```python
from flair.data import Sentence
from flair.models import SequenceTagger

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

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
```

This should print: 

```console
Sentence: "I love Berlin ." - 4 Tokens

The following NER tags are found: 

LOC-span [3]: "Berlin"
```

## Tutorials

We provide a set of quick tutorials to get you started with the library:

* [Tutorial 1: Basics](/resources/docs/TUTORIAL_1_BASICS.md)
* [Tutorial 2: Tagging your Text](/resources/docs/TUTORIAL_2_TAGGING.md)
* [Tutorial 3: Using Word Embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md)
* [Tutorial 4: Using BERT, ELMo, and Flair Embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)
* [Tutorial 5: Using Document Embeddings](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)
* [Tutorial 6: Loading your own Corpus](/resources/docs/TUTORIAL_6_CORPUS.md)
* [Tutorial 7: Training your own Models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
* [Tutorial 8: Optimizing your own Models](/resources/docs/TUTORIAL_8_MODEL_OPTIMIZATION.md)
* [Tutorial 9: Training your own Flair Embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)
 
The tutorials explain how the base NLP classes work, how you can load pre-trained models to tag your
text, how you can embed your text with different word or document embeddings, and how you can train your own 
language models, sequence labeling models, and text classification models. Let us know if anything is unclear.

There are also good third-party articles and posts that illustrate how to use Flair: 
* [How to build a text classifier with Flair](https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)
* [How to build a microservice with Flair and Flask](https://shekhargulati.com/2019/01/04/building-a-sentiment-analysis-python-microservice-with-flair-and-flask/)
* [A docker image for Flair](https://towardsdatascience.com/docker-image-for-nlp-5402c9a9069e)
* [Great overview of Flair functionality and how to use in Colab](https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/)
* [Visualisation tool for highlighting the extracted entities](https://github.com/lunayach/visNER)


## Citing Flair

Please cite the following paper when using Flair: 

```
@inproceedings{akbik2018coling,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {1638--1649},
  year      = {2018}
}
```

If you use the pooled version of the Flair embeddings (PooledFlairEmbeddings), please cite:

```
@inproceedings{akbik2019naacl,
  title={Pooled Contextualized Embeddings for Named Entity Recognition},
  author={Akbik, Alan and Bergmann, Tanja and Vollgraf, Roland},
  booktitle = {{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  pages     = {724–728},
  year      = {2019}
}
```

## Contact 

Please email your questions or comments to [Alan Akbik](http://alanakbik.github.io/).

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; 
start with our [contributor guidelines](CONTRIBUTING.md) and then 
check these [open issues](https://github.com/zalandoresearch/flair/issues) for specific tasks.

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit 
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around 
the code should hopefully be easy.

### Running unit tests locally

You need [Pipenv](https://pipenv.readthedocs.io/) for this:

```bash
pipenv install --dev && pipenv shell
pytest tests/
```

To run integration tests execute:
```bash
pytest --runintegration tests/
```
The integration tests will train small models.
Afterwards, the trained model will be loaded for prediction.

To also run slow tests, such as loading and using the embeddings provided by flair, you should execute:
```bash
pytest --runslow tests/
```

### Code Style

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black).
If your code is not formatted properly, travis will fail to build.

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.

If you want to set it up manually, install black via `pip install black`.
To reformat files execute `black .`.

## [License](/LICENSE)

The MIT License (MIT)

Flair is licensed under the following MIT license: The MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

