# HunFlair2 Tutorial 4: Customizing linking models

In this tutorial you'll find information on how to customize the entity linking models according to your needs.
As of now, fine-tuning the models is not supported.

## Customize dictionary

All linking models come with a pre-defined pairing of entity type and dictionary,
e.g. "Disease" mentions are linked by default to the [CTD Diseases](https://ctdbase.org/help/diseaseDetailHelp.jsp).
You can change the dictionary to which mentions are linked by following the steps below.
We'll be using the [Human Phenotype Ontology](https://hpo.jax.org/app/) in our example
(Download the `hp.json` file you find [here](https://hpo.jax.org/app/data/ontology) if you want to follow along).

First we load from the original data a python dictionary mapping names to concept identifiers

```python
import json
from collections import defaultdict
with open("hp.json") as fp:
    data = json.load(fp)

nodes = [n for n in data['graphs'][0]['nodes'] if n.get('type') == 'CLASS']
hpo = defaultdict(list)
for node in nodes:
    concept_id = node['id'].replace('http://purl.obolibrary.org/obo/', '')
    names = [node['lbl']] + [s['val'] for s in node.get('synonym', [])]
    for name in names:
        hpo[name].append(concept_id)  
```

Then we can convert this mapping into a dictionary that can be used by our linking model:

```python
from flair.datasets.entity_linking import (
    InMemoryEntityLinkingDictionary,
    EntityCandidate,
)

database_name="HPO"

candidates = [
    EntityCandidate(
        concept_id=ids[0],
        concept_name=name,
        additional_ids=ids[1:],
        database_name=database_name,
    )
    for name, ids in hpo.items()
]

dictionary =  InMemoryEntityLinkingDictionary(
    candidates=candidates, dataset_name=database_name
)
```

To use this dictionary you need to initialize a new linker model with it.
See the section below for that.

## Custom pre-trained model

You can initialize a new linker model with both a custom model and  custom dictionary (see section above) like this:

```python
pretrained_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
linker = EntityMentionLinker.build(
                pretrained_model,
                dictionary=dictionary,
                hybrid_search=False, 
                entity_type="disease",
            )
```

Omitting the `dictionary` parameter will load the default dictionary for the specified `entity_type`.

## Customizing Prediction Labels

In the default setup all linker models output their prediction into the same annotation category *link*.
To record the NEN annotation in separate categories, you can use the `pred_label_type` parameter of the
`predict()` method:

```python
gene_linker.predict(sentence, pred_label_type="my-genes")
disease_linker.predict(sentence, pred_label_type="my-diseases")

print("Diseases:")
for disease_tag in sentence.get_labels("my-diseases"):
    print(disease_tag)

print("\nGenes:")
for gene_tag in sentence.get_labels("my-genes"):
    print(gene_tag)
```

This will output:

```
Diseases:
Span[7:9]: "X-linked adrenoleukodystrophy" → MESH:D000326/name=Adrenoleukodystrophy (195.30780029296875)
Span[11:13]: "neurodegenerative disease" → MESH:D019636/name=Neurodegenerative Diseases (201.1804962158203)

Genes:
Span[4:5]: "ABCD1" → 215/name=ABCD1 (210.89810180664062)
```

Moreover, each linker has a pre-defined configuration specifying for which NER annotations it should compute
entity links:

```python
print(gene_linker.entity_label_types)
print(disease_linker.entity_label_types)
```

By default all models will use the *ner* annotation category and apply the linking algorithm for annotations
of the respective entity type:

```python
{'ner': {'gene'}}
{'ner': {'disease'}}
```

You can customize this by using the `entity_label_types` parameter of the `predict()` method:

```python
sentence = Sentence(
    "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
    "a neurodegenerative disease, which is exacerbated by exposure to high "
    "levels of mercury in mouse populations."
)

from flair.models import SequenceTagger

# Use disease ner tagger from HunFlair v1
hunflair1_tagger = SequenceTagger.load("hunflair-disease")
hunflair1_tagger.predict(sentence, label_name="my-diseases")

# Use the entity_label_types parameter in predict() to specify the annotation category
disease_linker.predict(sentence, entity_label_types="my-diseases")
```

If you are using annotated texts with more fine-granular NER annotations you are able to specify the
annotation category and tag type using a dictionary. For instance:

```python
gene_linker.predict(sentence, entity_label_types={"ner": {"gene": "protein"}})
```
