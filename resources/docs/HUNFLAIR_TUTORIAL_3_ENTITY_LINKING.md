# HunFlair Tutorial 3: Entity Linking

After adding Named Entity Recognition tags to your sentence, you can run Named Entity Linking on these annotations. 
```python
from flair.models.biomedical_entity_linking import BiomedicalEntityLinker
from flair.nn import Classifier
from flair.tokenization import SciSpacyTokenizer
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer())

ner_tagger = Classifier.load("hunflair-disease")
ner_tagger.predict(sentence)

nen_tagger = BiomedicalEntityLinker.load("disease")
nen_tagger.predict(sentence)

for tag in sentence.get_labels():
    print(tag)
```
This should print:
~~~
Span[0:2]: "Behavioral abnormalities" → Disease (0.6736)
Span[0:2]: "Behavioral abnormalities" →  behavior disorders - MESH:D001523 (0.9772)
Span[9:12]: "Fragile X Syndrome" → Disease (0.99)
Span[9:12]: "Fragile X Syndrome" →  fragile x syndrome - MESH:D005600 (1.0976)
~~~
The output contains both the NER disease annotations and their entity / concept identifiers according to 
a knowledge base or ontology. We have pre-configured combinations of models and dictionaries for 
"disease", "chemical" and "gene". 

You can also provide your own model and dictionary:
```python
from flair.models.biomedical_entity_linking import BiomedicalEntityLinker

nen_tagger = BiomedicalEntityLinker.load("name_or_path_to_your_model", dictionary_names_or_paths="name_or_path_to_your_dictionary")
nen_tagger = BiomedicalEntityLinker.load("path_to_custom_disease_model", dictionary_names_or_paths="disease")
````
You can use any combination of provided models, provided dictionaries and your own.
