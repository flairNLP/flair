# HunFlair Tutorial 3: Entity Linking

After adding Named Entity Recognition tags to your sentence, you can run Named Entity Linking on these annotations. 
```python
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer())

ner_tagger = MultiTagger.load("hunflair-disease")
ner_tagger.predict(sentence)

nen_tagger = MultiBiEncoderEntityLinker.load("disease")
nen_tagger.predict(sentence)

for tag in sentence.get_labels():
    print(tag)
```
This should print:
~~~
Disease [Behavioral abnormalities (1,2)] (0.6736)
Disease [Fragile X Syndrome (10,11,12)] (0.99)
MESH:D001523 (DO:DOID:150)  behavior disorders (0.98)
MESH:D005600 (DO:DOID:14261, OMIM:300624, OMIM:309548)  fragile x syndrome (1.1)
~~~
This output contains the NER disease annotations and it's entity linking annotations with ids from (often more than one) database.
We have preconfigured combinations of models and dictionaries for "disease", "chemical" and "gene". You can also provide your own model and dictionary:

```python
nen_tagger = MultiBiEncoderEntityLinker.load("name_or_path_to_your_model", dictionary_names_or_paths="name_or_path_to_your_dictionary")
nen_tagger = MultiBiEncoderEntityLinker.load("path_to_custom_disease_model", dictionary_names_or_paths="disease")
````
You can use any combination of provided models, provided dictionaries and your own.
