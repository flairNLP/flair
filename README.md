![alt text](resources/docs/flair_logo_2020_FINAL_day_dpi72.png#gh-light-mode-only)
![alt text](resources/docs/flair_logo_2020_FINAL_night_dpi72.png#gh-dark-mode-only)

[![PyPI version](https://badge.fury.io/py/flair.svg)](https://badge.fury.io/py/flair)
[![GitHub Issues](https://img.shields.io/github/issues/flairNLP/flair.svg)](https://github.com/flairNLP/flair/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

A very simple framework for **state-of-the-art NLP**. Developed by [Humboldt University of Berlin](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/) and friends.

---

Flair is:

* **A powerful NLP library.** Flair allows you to apply our state-of-the-art natural language processing (NLP)
models to your text, such as named entity recognition (NER), sentiment analysis, part-of-speech tagging (PoS),
  special support for [biomedical texts](/resources/docs/HUNFLAIR2.md),
 sense disambiguation and classification, with support for a rapidly growing number of languages.

* **A text embedding library.** Flair has simple interfaces that allow you to use and combine different word and
document embeddings, including our proposed [Flair embeddings](https://www.aclweb.org/anthology/C18-1139/) and various transformers.

* **A PyTorch NLP framework.** Our framework builds directly on [PyTorch](https://pytorch.org/), making it easy to
train your own models and experiment with new approaches using Flair embeddings and classes.

Now at [version 0.14.0](https://github.com/flairNLP/flair/releases)!


## State-of-the-Art Models

Flair ships with state-of-the-art models for a range of NLP tasks. For instance, check out our latest NER models:

| Language | Dataset | Flair | Best published | Model card & demo
|  ---  | ----------- | ---------------- | ------------- | ------------- |
| English | Conll-03 (4-class)   |  **94.09**  | *94.3 [(Yamada et al., 2020)](https://doi.org/10.18653/v1/2020.emnlp-main.523)* | [Flair English 4-class NER demo](https://huggingface.co/flair/ner-english-large)  |
| English | Ontonotes (18-class)  |  **90.93**  | *91.3 [(Yu et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair English 18-class NER demo](https://huggingface.co/flair/ner-english-ontonotes-large) |
| German  | Conll-03 (4-class)   |  **92.31**  | *90.3 [(Yu et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair German 4-class NER demo](https://huggingface.co/flair/ner-german-large)  |
| Dutch  | Conll-03  (4-class)  |  **95.25**  | *93.7 [(Yu et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Dutch 4-class NER demo](https://huggingface.co/flair/ner-dutch-large)  |
| Spanish  | Conll-03 (4-class)   |  **90.54** | *90.3 [(Yu et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Spanish 4-class NER demo](https://huggingface.co/flair/ner-spanish-large)  |

Many Flair sequence tagging models (named entity recognition, part-of-speech tagging etc.) are also hosted
on the [__🤗 Hugging Face model hub__](https://huggingface.co/models?library=flair&sort=downloads)! You can browse models, check detailed information on how they were trained, and even try each model out online!


## Quick Start

### Requirements and Installation

In your favorite virtual environment, simply do:

```
pip install flair
```

Flair requires Python 3.8+. 

### Example 1: Tag Entities in Text

Let's run **named entity recognition** (NER) over an example sentence. All you need to do is make a `Sentence`, load
a pre-trained model and use it to predict tags for the sentence:

```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
tagger = Classifier.load('ner')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

This should print:

```console
Sentence: "I love Berlin ." → ["Berlin"/LOC]
```

This means that "Berlin" was tagged as a **location entity** in this sentence. 

   * *to learn more about NER tagging in Flair, check out our [NER tutorial](https://flairnlp.github.io/docs/tutorial-basics/tagging-entities)!*


### Example 2: Detect Sentiment 

Let's run **sentiment analysis** over an example sentence to determine whether it is POSITIVE or NEGATIVE.
Same code as above, just a different model: 

```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
tagger = Classifier.load('sentiment')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

This should print:

```console
Sentence[4]: "I love Berlin ." → POSITIVE (0.9983)
```

This means that the sentence "I love Berlin" was tagged as having **POSITIVE** sentiment. 

   * *to learn more about sentiment analysis in Flair, check out our [sentiment analysis tutorial](https://flairnlp.github.io/docs/tutorial-basics/tagging-sentiment)!*

## Tutorials

On our new :fire: [**Flair documentation page**](https://flairnlp.github.io/docs/intro) you will find many tutorials to get you started!

In particular: 
- [Tutorial 1: Basic tagging](https://flairnlp.github.io/docs/category/tutorial-1-basic-tagging) → how to tag your text 
- [Tutorial 2: Training models](https://flairnlp.github.io/docs/category/tutorial-2-training-models) → how to train your own state-of-the-art NLP models 
- [Tutorial 3: Embeddings](https://flairnlp.github.io/docs/category/tutorial-3-embeddings) → how to produce embeddings for words and documents
- [Tutorial 4: Biomedical text](https://flairnlp.github.io/docs/category/tutorial-4-biomedical-text) → how to analyse biomedical text data

There is also a dedicated landing page for our [biomedical NER and datasets](/resources/docs/HUNFLAIR.md) with
installation instructions and tutorials.


## More Documentation

Another great place to start is the book [Natural Language Processing with Flair](https://www.amazon.com/Natural-Language-Processing-Flair-understanding/dp/1801072310)
and its accompanying [code repository](https://github.com/PacktPublishing/Natural-Language-Processing-with-Flair), though it was
written for an older version of Flair and some examples may no longer work.

There are also good third-party articles and posts that illustrate how to use Flair:
* [Training an NER model with Flair](https://medium.com/thecyphy/training-custom-ner-model-using-flair-df1f9ea9c762)
* [Training a text classifier with Flair](https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)
* [Zero and few-shot learning](https://towardsdatascience.com/zero-and-few-shot-learning-c08e145dc4ed) 
* [Visualisation tool for highlighting the extracted entities](https://github.com/lunayach/visNER)
* [Flair functionality and how to use in Colab](https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/)
* [Benchmarking NER algorithms](https://towardsdatascience.com/benchmark-ner-algorithm-d4ab01b2d4c3)
* [Clinical NLP](https://towardsdatascience.com/clinical-natural-language-processing-5c7b3d17e137)
* [How to build a microservice with Flair and Flask](https://shekhargulati.com/2019/01/04/building-a-sentiment-analysis-python-microservice-with-flair-and-flask/)
* [A docker image for Flair](https://towardsdatascience.com/docker-image-for-nlp-5402c9a9069e)
* [Practical approach of State-of-the-Art Flair in Named Entity Recognition](https://medium.com/analytics-vidhya/practical-approach-of-state-of-the-art-flair-in-named-entity-recognition-46a837e25e6b)
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
  title={{FLAIR}: An easy-to-use framework for state-of-the-art {NLP}},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}
```

If you use our new "FLERT" models or approach, please cite [this paper](https://arxiv.org/abs/2011.06993):

```
@misc{schweter2020flert,
    title={{FLERT}: Document-Level Features for Named Entity Recognition},
    author={Stefan Schweter and Alan Akbik},
    year={2020},
    eprint={2011.06993},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

If you use our TARS approach for few-shot and zero-shot learning, please cite [this paper](https://aclanthology.org/2020.coling-main.285/):

```
@inproceedings{halder2020coling,
  title={Task Aware Representation of Sentences for Generic Text Classification},
  author={Halder, Kishaloy and Akbik, Alan and Krapac, Josip and Vollgraf, Roland},
  booktitle = {{COLING} 2020, 28th International Conference on Computational Linguistics},
  year      = {2020}
}
```

## Contact

Please email your questions or comments to [Alan Akbik](http://alanakbik.github.io/).

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/flairNLP/flair/issues) for specific tasks.


## [License](/LICENSE)

The MIT License (MIT)

Flair is licensed under the following MIT license: The MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
