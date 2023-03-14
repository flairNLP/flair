# Tutorial 2.3: Entity Linking on Your Text

This is the **entity linking** tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. 

This tutorial contains the following parts:
* tagging and linking entities  
* understanding and accessing annotations (important!)
* tagging a whole text corpus
* list of all models in Flair

Let's get started!

## Tagging and Linking Entities

As of Flair 0.12 we ship an **experimental entity linker** trained on the [Zelda dataset](https://github.com/flairNLP/zelda). The linker does not only
tag entities, but also attempts to link each entity to the corresponding Wikipedia URL if one exists. 

To illustrate, let's use a short example text with two mentions of "Barcelona". The first refers to the football club
"FC Barcelona", the second to the city "Barcelona".

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('linker')

# make a sentence
sentence = Sentence('Bayern played against Barcelona. The match took place in Barcelona.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[12]: "Bayern played against Barcelona. The match took place in Barcelona." → ["Bayern"/FC_Bayern_Munich, "Barcelona"/FC_Barcelona, "Barcelona"/Barcelona]
```

As we can see, the linker can resolve what the two mentions of "Barcelona" refer to: 
- the first mention "Barcelona" is linked to "FC_Barcelona" 
- the second mention "Barcelona" is linked to "Barcelona"

Additionally, the mention "Bayern" is linked to "FC_Bayern_Munich", telling us that here the football club is meant.



## Understanding and Accessing Annotations (important!)

You can access each prediction individually using the `get_labels()` method. 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('linker')

# make a sentence
sentence = Sentence('Bayern played against Barcelona. The match took place in Barcelona.')

# predict NER tags
tagger.predict(sentence)
```

Use the `get_labels()` method to iterate over all predictions. Direct access each label's value (predicted tag)
and its confidence score.

```python
# Use the `get_labels()` method to iterate over all predictions. 
for label in sentence.get_labels():
    print(label)
    # print label value and score
    print(f'label.value is: "{label.value}"')
    print(f'label.score is: "{label.score}"')
```

This should print:

```console
Span[0:1]: "Bayern" → FC_Bayern_Munich (0.7778)
label.value is: "FC_Bayern_Munich"
label.score is: "0.7777503132820129"

Span[3:4]: "Barcelona" → FC_Barcelona (0.9983)
label.value is: "FC_Barcelona"
label.score is: "0.9983417987823486"

Span[10:11]: "Barcelona" → Barcelona (1.0)
label.value is: "Barcelona"
label.score is: "0.999983549118042"
```


## Tagging a Whole Text Corpus

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of `Sentence` objects to the `.predict()` method.

For instance, you can use the sentence splitter of segtok to split your text:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "Bayern played against Barcelona. The match took place in Barcelona."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = Classifier.load('linker')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

This should print: 

```console
Sentence[5]: "Bayern played against Barcelona." → ["Bayern"/FC_Bayern_Munich, "Barcelona"/FC_Barcelona]
Sentence[7]: "The match took place in Barcelona." → ["Barcelona"/Barcelona]
```

Using the `mini_batch_size` parameter of the `.predict()` method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.

## Next

Go back to the [tagging overview tutorial](/resources/docs/TUTORIAL_TAGGING_OVERVIEW.md) to check out other model types in Flair. Or learn how to train your own sentiment analysis model in our [text classifier training tutorial]/resources/docs/TUTORIAL_TRAINING_TEXT_CLASSIFIER.md).
