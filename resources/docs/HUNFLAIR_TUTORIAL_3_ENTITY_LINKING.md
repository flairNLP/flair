# HunFlair Tutorial 3: Entity Linking

Named Entity Linking requires that you ran Named Entity Recognition first.
```python
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer())

ner_tagger = MultiTagger.load("hunflair-disease")
```