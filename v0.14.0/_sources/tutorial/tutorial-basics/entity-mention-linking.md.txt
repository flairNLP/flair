# Using and creating entity mention linker

As of Flair 0.14 we ship the [entity mention linker](#flair.models.EntityMentionLinker) - the core framework behind the [Hunflair BioNEN approach](https://huggingface.co/hunflair)]. 
You can read more at the [Hunflair2 tutorials](project:../tutorial-hunflair2/overview.md)

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

ner_tagger = Classifier.load("hunflair2")
ner_tagger.predict(sentence)

nen_tagger = EntityMentionLinker.load("disease-linker")
nen_tagger.predict(sentence)

for tag in sentence.get_labels():
    print(tag)
```

```{note}
  Here we use the `disease-linker-no-ab3p` model, as it is the simplest model to run. You might get better results by using `disease-linker` instead,
  but that would require you to install `pyab3p` via `pip install pyab3p`.
  
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


## Example 2: Structured handling of predictions

After the predictions, the flair sentence has multiple labels added to the sentence object.
* Each NER prediction adds a span referenced by the `label_type` from the span tagger.
* Each NEL prediction adds one or more labels (up to `k`) to the respective span. Those have the `label_type` from the entity mention linker. 
* The NEL labels are ordered by their score. Depending on the exact implementation, it is possible that the order is ascending or descending, however the first one is always the best. 

Therefore, an example to extract the information to a dictionary that could be used for further processing is the following:

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

# top_k = 5 so that a span can have up to 5 labels assigned.
nen_tagger.predict(sentence, top_k=5)

result_mentions = []

for span in sentence.get_spans(ner_tagger.label_type):
    
    # basic information about the span that is tagged.
    span_data = {
        "start": span.start_position + sentence.start_position,
        "end": span.end_position + sentence.start_position,
        "text": span.text,
    }
    
    # add the ner label. We always have only one, so we can use `span.get_label(...)`
    span_data["ner_label"] = span.get_label(ner_tagger.label_type).value
    
    mentions_found = []
    
    # since `top_k` is larger than 1, we need to handle multiple nen labels. Therefore we use `span.get_labels(...)`
    for label in span.get_labels(nen_tagger.label_type):
        mentions_found.append({
            "id": label.value,
            "score": label.score,
        })
        
    # extract the most probable prediction if any prediction is found. 
    if mentions_found:
        span_data["nen_id"] = mentions_found[0]["id"]
    else:
        span_data["nen_id"] = None
        
    # add all found candidates with rating if you want to explore more than just the most probable prediction.
    span_data["mention_candidates"] = mentions_found
    
    result_mentions.append(span_data)

print(result_mentions)
```

```{note}
  If you need more than the extracted ids, you can use `nen_tagger.dictionary[span_data["nen_id"]]`
  to look up the [`flair.data.EntityCandidate`](#flair.data.EntityCandidate) which contains further information.
```