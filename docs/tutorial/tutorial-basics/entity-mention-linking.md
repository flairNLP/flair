# Using and creating entity mention linker

As of Flair 0.14 we ship the [entity mention linker](#flair.models.EntityMentionLinker) - the core framework behind the [Hunflair BioNEN aproach](https://huggingface.co/hunflair)]. 

## Example 1: Printing Entity linking outputs to console

To illustrate, let's use the example the hunflair models on a biomedical sentence:

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

nen_tagger = EntityMentionLinker.load("disease-linker-no-ab3p")
nen_tagger.predict(sentence)

for tag in sentence.get_labels():
    print(tag)
```

```{note}
  Here we use the `disease-linker-no-ab3p` model, as it is the simplest model to run. You might get better results by using `disease-linker` instead,
  but under the hood ab3p uses an executeable that is only compiled for linux and therefore won't run on every system.
  
  Analogously to `disease` there are also linker for `chemical`, `species` and `gene`
  all work with the `{entity_type}-linker` or `{entity_type}-linker-no-ab3p` naming-schema 
```


This should print:
```console
Span[4:5]: "ABCD1" → Gene (0.9509)
Span[7:11]: "X-linked adrenoleukodystrophy" → Disease (0.9872)
Span[7:11]: "X-linked adrenoleukodystrophy" → MESH:D000326/name=Adrenoleukodystrophy (195.30780029296875)
Span[13:15]: "neurodegenerative disease" → Disease (0.8988)
Span[13:15]: "neurodegenerative disease" → MESH:D019636/name=Neurodegenerative Diseases (201.1804962158203)
Span[29:30]: "mercury" → Chemical (0.9484)
Span[31:32]: "dolphin" → Species (0.8071)
```

As we can see, the huflair-ner model resolved entities of several types, however for the disease linker, only those of type disease were relevant: 
- "X-linked adrenoleukodystrophy" refers to the entity "[Adrenoleukodystrophy](https://id.nlm.nih.gov/mesh/D000326.html)"
- "neurodegenerative disease" refers to the "[Neurodegenerative Diseases](https://id.nlm.nih.gov/mesh/D019636.html)" 


## Example 2: Extracting Ids for programmatic usage

While printing all labels is a nice way to just display :

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

nen_tagger = EntityMentionLinker.load("disease-linker-no-ab3p")
nen_tagger.predict(sentence)

# TODO

```