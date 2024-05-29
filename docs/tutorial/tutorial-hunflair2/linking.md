# HunFlair2 - Tutorial 2: Entity Linking

[Part 1](project:./tagging.md) of the tutorial, showed how to use our pre-trained *HunFlair2* models to
tag biomedical entities in your text. However, documents from different biomedical (sub-) fields may use different
terms to refer to the exact same concept, e.g., “_tumor protein p53_”, “_tumor suppressor p53_”, “_TRP53_” are all
valid names for the gene “TP53” ([NCBI Gene:7157](https://www.ncbi.nlm.nih.gov/gene/7157)).
For improved integration and aggregation of entity mentions from multiple different documents linking / normalizing
the entities to standardized ontologies or knowledge bases is required.

## Linking with pre-trained HunFlair2 Models

After adding named entity recognition tags to your sentence, you can link the entities to standard ontologies
using distinct, type-specific linking models:

```python
from flair.models import EntityMentionLinker
from flair.nn import Classifier
from flair.data import Sentence

sentence = Sentence(
    "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
    "a neurodegenerative disease, which is exacerbated by exposure to high "
    "levels of mercury in mouse populations."
)

# Tag named entities in the text
ner_tagger = Classifier.load("hunflair2")
ner_tagger.predict(sentence)

# Load disease linker and perform disease linking
disease_linker = EntityMentionLinker.load("disease-linker")
disease_linker.predict(sentence)

# Load gene linker and perform gene linking
gene_linker = EntityMentionLinker.load("gene-linker")
gene_linker.predict(sentence)

# Load chemical linker and perform chemical linking
chemical_linker = EntityMentionLinker.load("chemical-linker")
chemical_linker.predict(sentence)

# Load species linker and perform species linking
species_linker = EntityMentionLinker.load("species-linker")
species_linker.predict(sentence)
```

```{note}
the ontologies and knowledge bases used are pre-processed the first time the normalisation is executed,
which might takes a certain amount of time. All further calls are then based on this pre-processing and run
much faster.
```

After running the code we can inspect and output the linked entities via:

```python
for tag in sentence.get_labels("link"):
    print(tag)
```

This should print:

```
Span[4:5]: "ABCD1" → 215/name=ABCD1 (210.89810180664062)
Span[7:9]: "X-linked adrenoleukodystrophy" → MESH:D000326/name=Adrenoleukodystrophy (195.30780029296875)
Span[11:13]: "neurodegenerative disease" → MESH:D019636/name=Neurodegenerative Diseases (201.1804962158203)
Span[23:24]: "mercury" → MESH:D008628/name=Mercury (220.39199829101562)
Span[25:26]: "mouse" → 10090/name=Mus musculus (213.6201934814453)
```

For each entity, the output contains both the NER mention annotations and their ontology identifiers to which
the mentions were mapped. Moreover, the official name of the entity in the ontology and the similarity score
of the entity mention and the ontology concept is given. For instance, the official name for the disease
"_X-linked adrenoleukodystrophy_" is adrenoleukodystrophy. The similarity scores are specific to entity type,
ontology and linking model used and can therefore only be compared and related to those using the exact same
setup.

## Overview of pre-trained Entity Linking Models

HunFlair2 comes with the following pre-trained linking models:

| Entity Type | Model Name        | Ontology / Dictionary                                      | Linking Algorithm / Base Model (Data Set)                                               |
| ----------- | ----------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Chemical    | `chemical-linker` | [CTD Chemicals](https://ctdbase.org/downloads/#allchems)   | [SapBERT (BC5CDR)](https://huggingface.co/dmis-lab/biosyn-sapbert-bc5cdr-chemical)      |
| Disease     | `disease-linker`  | [CTD Diseases](https://ctdbase.org/downloads/#alldiseases) | [SapBERT (NCBI Disease)](https://huggingface.co/dmis-lab/biosyn-sapbert-bc5cdr-disease) |
| Gene        | `gene-linker`     | [NCBI Gene (Human)](https://www.ncbi.nlm.nih.gov/gene)     | [SapBERT (BC2GN)](https://huggingface.co/dmis-lab/biosyn-sapbert-bc2gn)                 |
| Species     | `species-linker`  | [NCBI Taxonmy](https://www.ncbi.nlm.nih.gov/taxonomy)      | [SapBERT  (UMLS)](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) |

For detailed information concerning the different models and their integration please refer to [our paper](https://arxiv.org/abs/2402.12372).

If you wish to customize the models and dictionaries please refer to the [dedicated tutorial](project:./customize-linking.md).
