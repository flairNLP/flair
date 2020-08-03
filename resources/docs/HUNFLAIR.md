# HunFlair

<i>HunFlair</i> is a state-of-the-art NER tagger for biomedical texts. It comes with 
models for genes/proteins, chemicals, diseases, species and cell lines. <i>HunFlair</i> 
builds on pretrained domain-specific language models and outperforms other biomedical 
NER tools on unseen corpora. Furthermore, it contains harmonized versions of [31 biomedical 
NER data sets](HUNFLAIR_CORPORA.md).



<b>Content:</b> 
[Quick Start](#quick-start) | 
[BioNER-Tool Comparison](#comparison-to-other-biomedical-ner-tools) |
[Tutorials](#tutorials) | 
[Citing HunFlair](#citing-hunflair) 

## Quick Start

#### Requirements and Installation
<i>HunFlair</i> is based on Flair 0.6+ and Python 3.6+. 
If you do not have Python 3.6, install it first. [Here is how for Ubuntu 16.04](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
Then, in your favorite virtual environment, simply do:
```
pip install flair
```
You may also want to install [SciSpaCy](https://allenai.github.io/scispacy/) for improved pre-processing 
and tokenization of scientific / biomedical texts:
 ```
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
```
 
#### Example Usage
Let's run named entity recognition (NER) over an example sentence. All you need to do is 
make a Sentence, load a pre-trained model and use it to predict tags for the sentence:
```python
import flair

sentence = flair.data.Sentence("Behavioral Abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

tagger = flair.models.sequence_tagger_model.MultiTagger.load("hunflair")
tagger.predict(sentence)
```
Done! The Sentence now has entity annotations. Let's print the entities found by the tagger:
```python
for gene in sentence.get_spans("hunflair-gene"):
    print(gene)

for disease in sentence.get_spans("hunflair-disease"):
    print(disease)

for species in sentence.get_spans("hunflair-species"):
    print(species)
```
This should print:
~~~
Span [5]: "Fmr1"   [− Labels: Gene (0.6896)]
Span [1,2]: "Behavioral Abnormalities"   [− Labels: Disease (0.706)]
Span [10,11,12]: "Fragile X Syndrome"   [− Labels: Disease (0.9863)]
Span [7]: "Mouse"   [− Labels: Species (0.9517)]
~~~

## Comparison to other biomedical NER tools
Tools for biomedical NER are typically trained and evaluated on rather small gold standard data sets. 
However, they are applied "in the wild", i.e., to a much larger collection of texts, often varying in 
topic, entity distribution, genre (e.g. patents vs. scientific articles) and text type (e.g. abstract 
vs. full text), which can lead to severe drops in performance.

<i>HunFlair</i> outperforms other biomedical NER tools on corpora not used for training of neither HunFlair
or any of the competitor tools.

| Corpus         | Entity Type  | Misc<sup><sub>[1](#f1)</sub></sup>   | SciSpaCy | HUNER | HunFlair | 
| ---            | ---          | ---    | ---   | ---  | ---         |
| [CRAFT v4.0](https://github.com/UCDenver-ccp/CRAFT)     | Chemical     | 42.88 | 32.03 | 42.99 | *__59.83__* |
|                | Gene/Protein | 64.93 | 49.37 | 50.77 | *__73.51__* |
|                | Species      | 81.15 | 47.07 | 84.45 | *__85.04__* |
| [BioNLP 2013 CG](https://www.aclweb.org/anthology/W13-2008/) | Chemical     | 72.15 | 59.93 | 67.37 | *__81.82__* |
|                | Disease      | 55.64 | 56.27 | 55.32 | *__65.07__* |
|                | Gene/Protein | 68.97 | 65.80 | 71.22 | *__87.71__* |
|                | Species      | *__80.53__* | 58.47 | 67.84 | 76.41 |
| [Plant-Disease](http://gcancer.org/pdr/)  | Species      | 80.63 | 74.51 | 73.64 | *__83.44__*  |
<sub>All results are F1 scores using partial matching of predicted text offsets with the original char offsets 
of the gold standard data. We allow a shift by max one character.</sub>

<a name="f1">1</a>:  Misc displays the results of multiple taggers: 
[tmChem](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmchem/) for Chemical, 
[GNormPus](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/) for Gene and Species, and 
[DNorm](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/DNorm.html) for Disease


Here's how to [reproduce these numbers](XXX) using Flair. You can also find detailed evaluations and discussions in our paper.

## Tutorials
We provide a set of quick tutorials to get you started with HunFlair:
* [Tutorial 1: Tagging](HUNFLAIR_TUTORIAL_1_TAGGING.md)

## Citing HunFlair
Please cite the following paper when using HunFlair:
~~~
@article{weber2020hunflair,
  author    = {Weber, Leon and S{\"a}nger, Mario and M{\"u}nchmeyer, Jannes and Habibi, Maryam and Leser, Ulf and Akbik, Alan},
  title     = {HunFlair: An Easy-to-Use Tool for State-of-the-ArtBiomedical Named Entity Recognition},
  journal   = {},
  volume    = {},
  year      = {2020},
  ee        = {http://arxiv.org/abs/XXX}
}
~~~
