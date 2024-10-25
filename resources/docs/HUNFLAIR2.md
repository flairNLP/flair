# HunFlair2

*HunFlair2* is a state-of-the-art named entity tagger and linker for biomedical texts. It comes with
models for genes/proteins, chemicals, diseases, species and cell lines. *HunFlair2*
builds on pretrained domain-specific language models and outperforms other biomedical
NER tools on unseen corpora.

<b>Content:</b>
[Quick Start](#quick-start) |
[Tool Comparison](#comparison-to-other-biomedical-entity-extraction-tools) |
[Tutorials](#tutorials) |
[Citing HunFlair](#citing-hunflair2)

## Quick Start

#### Requirements and Installation
*HunFlair2* is based on Flair 0.14+ and Python 3.9+. If you do not have Python 3.9, install it first.
Then, in your favorite virtual environment, simply do:
```
pip install flair
```

#### Example 1: Biomedical NER 
Let's run named entity recognition (NER) over an example sentence. All you need to do is
make a Sentence, load a pre-trained model and use it to predict tags for the sentence:
```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence 
sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

# load biomedical NER tagger
tagger = Classifier.load("hunflair2")

# tag sentence
tagger.predict(sentence)
```
Done! The Sentence now has entity annotations. Let's print the entities found by the tagger:
```python
for entity in sentence.get_labels():
    print(entity)
```
This should print:
```console
Span[0:2]: "Behavioral abnormalities" → Disease (1.0)
Span[4:5]: "Fmr1" → Gene (1.0)
Span[6:7]: "Mouse" → Species (1.0)
Span[9:12]: "Fragile X Syndrome" → Disease (1.0)
```

#### Example 2: Biomedical NEN
For improved integration and aggregation from multiple different documents linking / normalizing the entities to 
standardized ontologies or knowledge bases is required. Let's perform entity normalization by using
specialized models per entity type:
```python
from flair.data import Sentence
from flair.models import EntityMentionLinker
from flair.nn import Classifier

# make a sentence
sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

# load biomedical NER tagger + predict entities
tagger = Classifier.load("hunflair2")
tagger.predict(sentence)

# load gene linker and perform normalization
gene_linker = EntityMentionLinker.load("gene-linker")
gene_linker.predict(sentence)

# load disease linker and perform normalization
disease_linker = EntityMentionLinker.load("disease-linker")
disease_linker.predict(sentence)

# load species linker and perform normalization
species_linker = EntityMentionLinker.load("species-linker")
species_linker.predict(sentence)
```
**Note**, the ontologies and knowledge bases used are pre-processed the first time the normalisation is executed, 
which might takes a certain amount of time. All further calls are then based on this pre-processing and run 
much faster.

Done! The Sentence now has entity normalizations. Let's print the entity identifiers found by the linkers:
```python
for entity in sentence.get_labels("link"):
    print(entity)
```
This should print:
```console
Span[0:2]: "Behavioral abnormalities" → MESH:D001523/name=Mental Disorders (197.9467010498047)
Span[4:5]: "Fmr1" → 108684022/name=FRAXA (219.9510040283203)
Span[6:7]: "Mouse" → 10090/name=Mus musculus (213.6201934814453)
Span[9:12]: "Fragile X Syndrome" → MESH:D005600/name=Fragile X Syndrome (193.7115020751953)
```

## Comparison to other biomedical entity extraction tools
Tools for biomedical entity extraction are typically trained and evaluated on single, rather small gold standard 
data sets.  However, they are applied "in the wild" to a much larger collection of texts, often varying in
topic, entity distribution, genre (e.g. patents vs. scientific articles) and text type (e.g. abstract
vs. full text), which can lead to severe drops in performance.

*HunFlair2* outperforms other biomedical entity extraction tools on corpora not used for training of neither 
*HunFlair2* or any of the competitor tools.

| Corpus                                                                                       | Entity Type | BENT  | BERN2 | PubTator Central | SciSpacy | HunFlair    |
|----------------------------------------------------------------------------------------------|-------------|-------|-------|------------------|----------|-------------|
| [MedMentions](https://github.com/chanzuckerberg/MedMentions)                                 | Chemical    | 40.90 | 41.79 | 31.28            | 34.95    | *__51.17__* |
|                                                                                              | Disease     | 45.94 | 47.33 | 41.11            | 40.78    | *__57.27__* |
| [tmVar (v3)](https://github.com/ncbi/tmVar3?tab=readme-ov-file)                              | Gene        | 0.54  | 43.96 | *__86.02__*      | -        | 76.75       |
| [BioID](https://biocreative.bioinformatics.udel.edu/media/store/files/2018/BC6_track1_1.pdf) | Species     | 10.35 | 14.35 | *__58.90__*      | 37.14    | 49.66       |
|||||
| Average                                                                                      | All         | 24.43 | 36.86 | 54.33            | 37.61    | *__58.79__* |

<sub>All results are F1 scores highlighting end-to-end performance, i.e., named entity recognition and normalization,
using partial matching of predicted text offsets with the original char offsets of the gold standard data. 
We allow a shift by max one character.</sub>

You can find detailed evaluations and discussions in [our paper](https://arxiv.org/abs/2402.12372).

## Tutorials
We provide a set of quick tutorials to get you started with *HunFlair2*:
* [Tutorial 1: Tagging biomedical named entities](HUNFLAIR2_TUTORIAL_1_TAGGING.md)
* [Tutorial 2: Linking biomedical named entities](HUNFLAIR2_TUTORIAL_2_LINKING.md)
* [Tutorial 3: Training NER models](HUNFLAIR2_TUTORIAL_3_TRAINING_NER.md)
* [Tutorial 4: Customizing linking](HUNFLAIR2_TUTORIAL_4_CUSTOMIZE_LINKING.md)

## Citing HunFlair2
Please cite the following paper when using *HunFlair2*:
~~~
@article{sanger2024hunflair2,
  title={HunFlair2 in a cross-corpus evaluation of biomedical named entity recognition and normalization tools},
  author={S{\"a}nger, Mario and Garda, Samuele and Wang, Xing David and Weber-Genzel, Leon and Droop, Pia and Fuchs, Benedikt and Akbik, Alan and Leser, Ulf},
  journal={arXiv preprint arXiv:2402.12372},
  year={2024}
}
~~~
