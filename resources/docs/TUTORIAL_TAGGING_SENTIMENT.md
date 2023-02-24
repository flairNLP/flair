# Tutorial 2.2: Sentiment Analysis on Your Text

This is the **sentiment analysis** tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. 

This tutorial contains the following parts:
* tagging sentiment  
* understanding and accessing annotations (important!)
* tagging a whole text corpus
* list of all models in Flair

Let's get started!

## Tagging Sentiment

Our standard sentiment analysis model uses distilBERT embeddings and was trained over a mix of corpora, notably
the Amazon review corpus, and can thus handle a variety of domains and language.

Let's use an example sentence:

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('sentiment')

# make a sentence
sentence = Sentence('This movie is not at all bad.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[8]: "This movie is not at all bad." → POSITIVE (0.9929)
```

Showing us that the sentence overall is tagged to be of POSITIVE sentiment. 

## Tagging Sentiment - Fast

We also offer an RNN-based variant which is faster but less accurate. Use it like this: 


```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('sentiment-fast')

# make a sentence
sentence = Sentence('This movie is very bad.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[6]: "This movie is very bad." → NEGATIVE (0.9999)
```

This indicates that the sentence is of NEGATIVE sentiment. As you can see, its the same code as above, just loading the
'**sentiment-fast**' model instead of '**sentiment**'.


## Understanding and Accessing Annotations (important!)

You can access each prediction individually using the `get_labels()` method. 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('sentiment')

# make a sentence
sentence = Sentence('This movie is not at all bad.')

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

Since there is only one prediction, this should print:

```console
Sentence[8]: "This movie is not at all bad." → POSITIVE (0.9929)
label.value is: "POSITIVE"
label.score is: "0.9928739070892334"
```


## Tagging a Whole Text Corpus

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of `Sentence` objects to the `.predict()` method.

For instance, you can use the sentence splitter of segtok to split your text:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "I first thought it was great. Then I realized it's terrible. But I came to the conclusion that it's great."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = Classifier.load('sentiment')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

This should print: 

```console
Sentence[7]: "I first thought it was great." → POSITIVE (0.6549)
Sentence[7]: "Then I realized it's terrible." → NEGATIVE (1.0)
Sentence[11]: "But I came to the conclusion that it's great." → POSITIVE (0.9846)
```

Using the `mini_batch_size` parameter of the `.predict()` method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.

### List of Sentiment Models

We end this section with a list of all models we currently ship with Flair:

| ID | Language | Task | Training Dataset | Accuracy |
| ------------- | ---- | ------------- |------------- |------------- |
| 'sentiment' | English | detecting positive and negative sentiment (transformer-based) | movie and product reviews |  **98.87** |
| 'sentiment-fast' | English | detecting positive and negative sentiment (RNN-based) | movie and product reviews |  **96.83**|
| 'de-offensive-language' | German | detecting offensive language | [GermEval 2018 Task 1](https://projects.fzai.h-da.de/iggsa/projekt/) |  **75.71** (Macro F1) |


## Next

Go back to the [tagging overview tutorial](/resources/docs/TUTORIAL_TAGGING_OVERVIEW.md) to check out other model types in Flair. Or learn how to train your own NER model in our training tutorial.
