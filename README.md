![alt text](resources/docs/flair_logo_2020.png)

[![PyPI version](https://badge.fury.io/py/flair.svg)](https://badge.fury.io/py/flair)
[![GitHub Issues](https://img.shields.io/github/issues/flairNLP/flair.svg)](https://github.com/flairNLP/flair/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

A very simple framework for **state-of-the-art NLP**. Developed by [Humboldt University of Berlin](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/) and friends.

---

Flair is:

* **A powerful NLP library.** Flair allows you to apply our state-of-the-art natural language processing (NLP)
models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS), 
  special support for [biomedical data](/resources/docs/HUNFLAIR.md),
 sense disambiguation and classification, with support for a rapidly growing number of languages.

* **A text embedding library.** Flair has simple interfaces that allow you to use and combine different word and
document embeddings, including our proposed **[Flair embeddings](https://www.aclweb.org/anthology/C18-1139/)**, BERT embeddings and ELMo embeddings.

* **A PyTorch NLP framework.** Our framework builds directly on [PyTorch](https://pytorch.org/), making it easy to
train your own models and experiment with new approaches using Flair embeddings and classes.

Now at [version 0.8](https://github.com/flairNLP/flair/releases)!

## State-of-the-Art Models

Flair ships with state-of-the-art models for a range of NLP tasks. For instance, check out our latest NER models: 

| Language | Dataset | Flair | Best published | Model card & demo
|  ---  | ----------- | ---------------- | ------------- | ------------- |
| English | Conll-03 (4-class)   |  **94.09**  | *94.3 [(Yamada et al., 2018)](https://doi.org/10.18653/v1/2020.emnlp-main.523)* | [Flair English 4-class NER demo](https://huggingface.co/flair/ner-english-large)  |
| English | Ontonotes (18-class)  |  **90.93**  | *91,3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair English 18-class NER demo](https://huggingface.co/flair/ner-english-ontonotes-large) |
| German  | Conll-03 (4-class)   |  **92,31**  | *90.3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair German 4-class NER demo](https://huggingface.co/flair/ner-german-large)  |
| Dutch  | Conll-03  (4-class)  |  **95,25**  | *93.7 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Dutch 4-class NER demo](https://huggingface.co/flair/ner-dutch-large)  |
| Spanish  | Conll-03 (4-class)   |  **90,54** | *90.3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Spanish 18-class NER demo](https://huggingface.co/flair/ner-spanish-large)  |

**New:** Most Flair sequence tagging models (named entity recognition, part-of-speech tagging etc.) are now hosted 
on the [__🤗 HuggingFace model hub__](https://huggingface.co/models?filter=flair)! You can browse models, check detailed information on how they were trained, and even try each model out online!


## Quick Start

### Requirements and Installation

The project is based on PyTorch 1.5+ and Python 3.6+, because method signatures and type hints are beautiful.
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

Span [3]: "Berlin"   [− Labels: LOC (0.9992)]
```

## Tutorials

We provide a set of quick tutorials to get you started with the library:

* [Tutorial 1: Basics](/resources/docs/TUTORIAL_1_BASICS.md)
* [Tutorial 2: Tagging your Text](/resources/docs/TUTORIAL_2_TAGGING.md)
* [Tutorial 3: Embedding Words](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md)
* [Tutorial 4: List of All Word Embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)
* [Tutorial 5: Embedding Documents](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)
* [Tutorial 6: Loading a Dataset](/resources/docs/TUTORIAL_6_CORPUS.md)
* [Tutorial 7: Training a Model](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
* [Tutorial 8: Training your own Flair Embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)
* [Tutorial 9: Training a Zero Shot Text Classifier (TARS)](/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)

The tutorials explain how the base NLP classes work, how you can load pre-trained models to tag your
text, how you can embed your text with different word or document embeddings, and how you can train your own
language models, sequence labeling models, and text classification models. Let us know if anything is unclear.

There is also a dedicated landing page for our **[biomedical NER and datasets](/resources/docs/HUNFLAIR.md)** with
installation instructions and tutorials.

There are also good third-party articles and posts that illustrate how to use Flair:
* [How to build a text classifier with Flair](https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)
* [How to build a microservice with Flair and Flask](https://shekhargulati.com/2019/01/04/building-a-sentiment-analysis-python-microservice-with-flair-and-flask/)
* [A docker image for Flair](https://towardsdatascience.com/docker-image-for-nlp-5402c9a9069e)
* [Great overview of Flair functionality and how to use in Colab](https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/)
* [Visualisation tool for highlighting the extracted entities](https://github.com/lunayach/visNER)
* [Practical approach of State-of-the-Art Flair in Named Entity Recognition](https://medium.com/analytics-vidhya/practical-approach-of-state-of-the-art-flair-in-named-entity-recognition-46a837e25e6b)
* [Benchmarking NER algorithms](https://towardsdatascience.com/benchmark-ner-algorithm-d4ab01b2d4c3)
* [Training a Flair text classifier on Google Cloud Platform (GCP) and serving predictions on GCP](https://github.com/robinvanschaik/flair-on-gcp)
* [Model Interpretability for transformer-based Flair models](https://github.com/robinvanschaik/interpret-flair)

## Citing Flair

Please cite [the following paper](https://www.aclweb.org/anthology/C18-1139/) when using Flair embeddings:

```
@inproceedings{akbik2018coling,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {1638--1649},
  year      = {2018}
}
```

If you use the Flair framework for your experiments, please cite [this paper](https://www.aclweb.org/anthology/papers/N/N19/N19-4010/):

```
@inproceedings{akbik2019flair,
  title={FLAIR: An easy-to-use framework for state-of-the-art NLP},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}
```

If you use the pooled version of the Flair embeddings (PooledFlairEmbeddings), please cite [this paper](https://www.aclweb.org/anthology/papers/N/N19/N19-1078/):

```
@inproceedings{akbik2019naacl,
  title={Pooled Contextualized Embeddings for Named Entity Recognition},
  author={Akbik, Alan and Bergmann, Tanja and Vollgraf, Roland},
  booktitle = {{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  pages     = {724–728},
  year      = {2019}
}
```

If you use our new "FLERT" models or approach, please cite [this paper](https://arxiv.org/abs/2011.06993):

```
@misc{schweter2020flert,
    title={FLERT: Document-Level Features for Named Entity Recognition},
    author={Stefan Schweter and Alan Akbik},
    year={2020},
    eprint={2011.06993},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
```

## Contact

Please email your questions or comments to [Alan Akbik](http://alanakbik.github.io/).

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/flairNLP/flair/issues) for specific tasks.

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

## [License](/LICENSE)

The MIT License (MIT)

Flair is licensed under the following MIT license: The MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
