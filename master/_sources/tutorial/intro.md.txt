---
sidebar_position: 1
---

(getting_started)=

# Quick Start

Let's discover **Flair in less than 5 minutes**.

## Requirements and Installation

In your favorite virtual environment, simply do:

```
pip install flair
```

Flair requires Python 3.9+. 

## Example 1: Tag Entities in Text

Let's run **named entity recognition**  (NER) over the following example sentence: "_I love Berlin and New York._"

Our goal is to identify names in this sentence, and their types.

To do this, all you need is to make a [`Sentence`](#flair.data.Sentence) for this text, load a pre-trained model and use it to predict tags for the sentence:


```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('I love Berlin and New York.')

# load the NER tagger
tagger = Classifier.load('ner')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

This should print:

```console
Sentence[7]: "I love Berlin and New York." → ["Berlin"/LOC, "New York"/LOC]
```

The output shows that both "Berlin" and "New York" were tagged as **location entities** (LOC) in this sentence.


## Example 2: Detect Sentiment 

Let's run **sentiment analysis** over the same sentence to determine whether it is POSITIVE or NEGATIVE.

You can do this with essentially the same code as above. Just instead of loading the 'ner' model, you now load the 'sentiment' model:


```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('I love Berlin and New York.')

# load the sentiment tagger
tagger = Classifier.load('sentiment')

# run sentiment analysis over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)

```

This should print:

```console
Sentence[7]: "I love Berlin and New York." → POSITIVE (0.9982)
```

The output shows that the sentence "_I love Berlin and New York._" was tagged as having **POSITIVE** sentiment. 


## Summary

Congrats, you now know how to use Flair to find entities and detect sentiment!