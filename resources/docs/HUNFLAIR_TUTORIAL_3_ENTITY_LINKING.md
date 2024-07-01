# HunFlair Tutorial 3: Entity Linking

After adding named entity recognition tags to your sentence, you can run named entity linking on these annotations.

```python
from flair.models import EntityMentionLinker
from flair.nn import Classifier
from flair.tokenization import SciSpacyTokenizer
from flair.data import Sentence

sentence = Sentence(
    "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
    "a neurodegenerative disease, which is exacerbated by exposure to high "
    "levels of mercury in dolphin populations.",
    use_tokenizer=SciSpacyTokenizer()
)

ner_tagger = Classifier.load("hunflair")
ner_tagger.predict(sentence)

nen_tagger = EntityMentionLinker.load("disease-linker")
nen_tagger.predict(sentence)

nen_tagger = EntityMentionLinker.load("gene-linker")
nen_tagger.predict(sentence)

nen_tagger = EntityMentionLinker.load("chemical-linker")
nen_tagger.predict(sentence)

nen_tagger = EntityMentionLinker.load("species-linker")
nen_tagger.predict(sentence)

for tag in sentence.get_labels():
    print(tag)
```

This should print:

```
Span[4:5]: "ABCD1" → Gene (0.9575)
Span[4:5]: "ABCD1" →  abcd1 - NCBI-GENE-HUMAN:215 (14.5503)
Span[7:11]: "X-linked adrenoleukodystrophy" → Disease (0.9867)
Span[7:11]: "X-linked adrenoleukodystrophy" →  x linked adrenoleukodystrophy - CTD-DISEASES:MESH:D000326 (13.9717)
Span[13:15]: "neurodegenerative disease" → Disease (0.8865)
Span[13:15]: "neurodegenerative disease" →  neurodegenerative disease - CTD-DISEASES:MESH:D019636 (14.2779)
Span[25:26]: "mercury" → Chemical (0.9456)
Span[25:26]: "mercury" →  mercury - CTD-CHEMICALS:MESH:D008628 (14.9185)
Span[27:28]: "dolphin" → Species (0.8082)
Span[27:28]: "dolphin" →  marine dolphins - NCBI-TAXONOMY:9726 (14.473)
```

The output contains both the NER disease annotations and their entity / concept identifiers according to
a knowledge base or ontology. We have pre-configured combinations of models and dictionaries for
"disease", "chemical" and "gene".

You can also provide your own model and dictionary:

```python
from flair.models import EntityMentionLinker

nen_tagger = EntityMentionLinker.build("name_or_path_to_your_model",
                                       dictionary_names_or_path="name_or_path_to_your_dictionary")
nen_tagger = EntityMentionLinker.build("path_to_custom_disease_model", dictionary_names_or_path="disease")
```

You can use any combination of provided models, provided dictionaries and your own.
