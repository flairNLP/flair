# Tutorial 2.5: Relation Extraction on Your Text

This is the **relation extraction** tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. 

This tutorial contains the following parts:
* tagging relations  
* understanding and accessing annotations (important!)
* tagging a whole text corpus

Let's get started!

## Tagging Relations

Relations hold between two entities. For instance, a text like "George was born in Washington"
names two entities and also expresses that there is a born_in relationship between
both.

We added an experimental relation extraction model trained over a modified version of TACRED.
You must use this model together with an entity tagger. Here is an example:

```python
from flair.data import Sentence
from flair.nn import Classifier

# 1. make example sentence
sentence = Sentence("George was born in Washington")

# 2. load entity tagger and predict entities
tagger = Classifier.load('ner-fast')
tagger.predict(sentence)

# check which named entities have been found in the sentence
entities = sentence.get_labels('ner')
for entity in entities:
    print(entity)

# 3. load relation extractor
extractor = Classifier.load('relations')

# predict relations
extractor.predict(sentence)

# check which relations have been found
relations = sentence.get_labels('relation')
for relation in relations:
    print(relation)

# Use the `get_labels()` method with parameter 'relation' to iterate over all relation predictions. 
for label in sentence.get_labels('relation'):
    print(label)
```

This should print:

```console
Span[0:1]: "George" → PER (0.9971)
Span[4:5]: "Washington" → LOC (0.9847)

Relation[0:1][4:5]: "George -> Washington" → born_in (1.0)
```

Indicating that a **born_in** relationship holds between "George" and "Washington"!


## Understanding and Accessing Annotations (important!)

As the example above shows, you can access each prediction individually using the `get_labels()` method. 
However, you should pass either 'ner' to get the NER labels, or 'relation' to get the relation labels.

If you want to iterate over the relations and directly access the predicted label or the score, 
do the following:

```python
# Use the `get_labels()` method with parameter 'relation' to iterate over all relation predictions. 
for label in sentence.get_labels('relation'):
    print(label)
    # print label value and score
    print(f'label.value is: "{label.value}"')
    print(f'label.score is: "{label.score}"')
    print(f'subject of the relation is: "{label.data_point.first}"')
    print(f'object of the relation is: "{label.data_point.second}"')
```

Since there is only one prediction, this should print:

```console
Relation[0:1][4:5]: "George -> Washington" → born_in (1.0)

label.value is: "born_in"
label.score is: "0.9999638795852661"

subject of the relation is: "Span[0:1]: "George" → PER (0.9971)"
object of the relation is: "Span[4:5]: "Washington" → LOC (0.9847)"
```


## Tagging a Whole Text Corpus

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of `Sentence` objects to the `.predict()` method.

For instance, you can use the sentence splitter of segtok to split your text:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "Dirk was born in Essen. But Dirk now lives in Gelsenkirchen."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
entity_tagger = Classifier.load('ner')
relation_extractor = Classifier.load('relations')

entity_tagger.predict(sentences)
relation_extractor.predict(sentences)

# iterate through sentences and print predicted relation labels
for sentence in sentences:
    print(sentence.get_labels("relation"))
```

This should print: 

```console
['Relation[0:1][4:5]: "Dirk -> Essen"'/'born_in' (1.0)]
['Relation[1:2][5:6]: "Dirk -> Gelsenkirchen"'/'lived_in' (0.9997)]
```

Using the `mini_batch_size` parameter of the `.predict()` method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.

## Next

Go back to the [tagging overview tutorial](/resources/docs/TUTORIAL_TAGGING_OVERVIEW.md) to check out other model types in Flair. Or learn how to train your own NER model in our training tutorial.
