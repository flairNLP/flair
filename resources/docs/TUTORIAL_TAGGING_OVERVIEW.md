# Tutorial 2: Tagging your Text

This is part 2 of the tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. 

This tutorial consists of several chapters, one for each type of model: 

* [Tutorial 2.1: How to use **Named Entity Recognition** models](/resources/docs/TUTORIAL_TAGGING_NER.md) 
* [Tutorial 2.2: How to use **Sentiment Analysis** models](/resources/docs/TUTORIAL_TAGGING_SENTIMENT.md)  
* [Tutorial 2.3: How to use **Entity Linking** models](/resources/docs/TUTORIAL_TAGGING_LINKING.md)  
* [Tutorial 2.4: How to use **Part-of-Speech Tagging** models](/resources/docs/TUTORIAL_TAGGING_POS.md)  
* [Tutorial 2.5: How to use **Relation Extraction** models](/resources/docs/TUTORIAL_TAGGING_RELATIONS.md)  
* [Tutorial 2.6: **Other crazy models** we ship with Flair](/resources/docs/TUTORIAL_TAGGING_CIRCUS.md)  

Click on any of these above links to look into the respectice tutorial chapters.

## Important for All Taggers: How Annotations Work

Let's use our standard NER example to illustrate how annotations work: 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('ner')

# make a sentence
sentence = Sentence('George Washington went to Washington.')

# predict NER tags
tagger.predict(sentence)

# print the sentence with the tags
print(sentence)
```

This should print:
```console
Sentence: "George Washington went to Washington ." → ["George Washington"/PER, "Washington"/LOC]
```

Showing us that two entities are labeled in this sentence: "George Washington" as PER (person) and "Washington"
as LOC (location.)

A common question that gets asked is **how to access these predictions directly**. You can do this by using
the `get_labels()` method to iterate over all predictions:

```python
for label in sentence.get_labels():
    print(label)
```
This should print the two NER predictions:

```console
Span[0:2]: "George Washington" → PER (0.9989)
Span[4:5]: "Washington" → LOC (0.9942)
```

As you can see, each entity is printed, together with the predicted class. 
The confidence of the prediction is indicated as a score in brackets.

For each prediction, you can even **directly access** the label value, it's score and the entity text:  

```python
# iterate over all labels in the sentence
for label in sentence.get_labels():
    # print label value and score
    print(f'label.value is: "{label.value}"')
    print(f'label.score is: "{label.score}"')
    # access the data point to which label attaches and print its text
    print(f'the text of label.data_point is: "{label.data_point.text}"\n')
```

This should print: 
```console
label.value is: "PER"
label.score is: "0.998886227607727"
the text of label.data_point is: "George Washington"

label.value is: "LOC"
label.score is: "0.9942097663879395"
the text of label.data_point is: "Washington"
```



## Next

You can check out all tutorial chapters listed above. We recommend starting with [Tutorial 2.1: How to use **Named Entity Recognition** models](/resources/docs/TUTORIAL_TAGGING_NER.md).
