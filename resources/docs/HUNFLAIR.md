# HunFlair

*HunFlair* is a state-of-the-art NER tagger for biomedical texts. It comes with 
models for genes/proteins, chemicals, diseases, species and cell lines. *HunFlair* 
builds on pretrained domain-specific language models and outperforms other biomedical 
NER tools on unseen corpora. Furthermore, it contains harmonized versions of [31 biomedical 
NER data sets](HUNFLAIR_CORPORA.md) and comes with a Flair language model ("pubmed-X") and
FastText embeddings ("pubmed") that were trained on roughly 3 million full texts and about
25 million abstracts from the biomedical domain.

<b>Content:</b> 
[Quick Start](#quick-start) | 
[BioNER-Tool Comparison](#comparison-to-other-biomedical-ner-tools) |
[Tutorials](#tutorials) | 
[Citing HunFlair](#citing-hunflair) 

## Quick Start

#### Requirements and Installation
*HunFlair* is based on Flair 0.6+ and Python 3.6+. 
If you do not have Python 3.6, install it first. [Here is how for Ubuntu 16.04](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
Then, in your favorite virtual environment, simply do:
```
pip install flair
```
Furthermore, we recommend to install [SciSpaCy](https://allenai.github.io/scispacy/) for improved pre-processing 
and tokenization of scientific / biomedical texts:
 ```
pip install scispacy==0.2.5
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
```
 
#### Example Usage
Let's run named entity recognition (NER) over an example sentence. All you need to do is 
make a Sentence, load a pre-trained model and use it to predict tags for the sentence:
```python
from flair.data import Sentence
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer

# make a sentence and tokenize with SciSpaCy
sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome",
                    use_tokenizer=SciSpacyTokenizer())

# load biomedical tagger
tagger = MultiTagger.load("hunflair")

# tag sentence
tagger.predict(sentence)
```
Done! The Sentence now has entity annotations. Let's print the entities found by the tagger:
```python
for annotation_layer in sentence.annotation_layers.keys():
    for entity in sentence.get_spans(annotation_layer):
        print(entity)
```
This should print:
~~~
Span[0:2]: "Behavioral abnormalities" → Disease (0.6736)
Span[9:12]: "Fragile X Syndrome" → Disease (0.99)
Span[4:5]: "Fmr1" → Gene (0.838)
Span[6:7]: "Mouse" → Species (0.9979)
~~~

## Comparison to other biomedical NER tools
Tools for biomedical NER are typically trained and evaluated on rather small gold standard data sets. 
However, they are applied "in the wild" to a much larger collection of texts, often varying in 
topic, entity distribution, genre (e.g. patents vs. scientific articles) and text type (e.g. abstract 
vs. full text), which can lead to severe drops in performance.

*HunFlair* outperforms other biomedical NER tools on corpora not used for training of neither *HunFlair*
or any of the competitor tools.

| Corpus         | Entity Type  | Misc<sup><sub>[1](#f1)</sub></sup>   | SciSpaCy | HUNER | HunFlair | 
| ---            | ---          | ---    | ---   | ---  | ---         |
| [CRAFT v4.0](https://github.com/UCDenver-ccp/CRAFT)     | Chemical     | 42.88 | 35.73 | 42.99 | *__59.83__* |
|                | Gene/Protein | 64.93 | 47.76 | 50.77 | *__73.51__* |
|                | Species      | 81.15 | 54.21 | 84.45 | *__85.04__* |
| [BioNLP 2013 CG](https://www.aclweb.org/anthology/W13-2008/) | Chemical     | 72.15 | 58.43 | 67.37 | *__81.82__* |
|                | Disease      | 55.64 | 56.48 | 55.32 | *__65.07__* |
|                | Gene/Protein | 68.97 | 66.18 | 71.22 | *__87.71__* |
|                | Species      | *__80.53__* | 57.11 | 67.84 | 76.41 |
| [Plant-Disease](http://gcancer.org/pdr/)  | Species      | 80.63 | 75.90 | 73.64 | *__83.44__*  |

<sub>All results are F1 scores using partial matching of predicted text offsets with the original char offsets 
of the gold standard data. We allow a shift by max one character.</sub>

<sub><a name="f1">1</a>:  Misc displays the results of multiple taggers: 
[tmChem](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmchem/) for Chemical, 
[GNormPus](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/) for Gene and Species, and 
[DNorm](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/DNorm.html) for Disease
</sub>

Here's how to [reproduce these numbers](HUNFLAIR_EXPERIMENTS.md) using Flair. 
You can find detailed evaluations and discussions in [our paper](https://arxiv.org/abs/2008.07347).

## Tutorials
We provide a set of quick tutorials to get you started with *HunFlair*:
* [Tutorial 1: Tagging](HUNFLAIR_TUTORIAL_1_TAGGING.md)
* [Tutorial 2: Training biomedical NER models](HUNFLAIR_TUTORIAL_2_TRAINING.md)

## Citing HunFlair
Please cite the following paper when using *HunFlair*:
~~~
@article{weber2020hunflair,
    title={HunFlair: An Easy-to-Use Tool for State-of-the-Art Biomedical Named Entity Recognition},
    author={Weber, Leon and S{\"a}nger, Mario and M{\"u}nchmeyer, Jannes  and Habibi, Maryam and Leser, Ulf and Akbik, Alan},
    journal={arXiv preprint arXiv:2008.07347},
    year={2020}
}
~~~
