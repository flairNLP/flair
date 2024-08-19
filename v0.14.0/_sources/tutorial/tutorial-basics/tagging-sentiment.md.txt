# Tagging sentiment

This tutorials shows you how to do sentiment analysis in Flair.

## Tagging sentiment with our standard model

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

## Tagging sentiment with our fast model

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


### List of Sentiment Models

We end this section with a list of all models we currently ship with Flair:

| ID | Language | Task | Training Dataset | Accuracy |
| ------------- | ---- | ------------- |------------- |------------- |
| 'sentiment' | English | detecting positive and negative sentiment (transformer-based) | movie and product reviews |  **98.87** |
| 'sentiment-fast' | English | detecting positive and negative sentiment (RNN-based) | movie and product reviews |  **96.83**|
| 'de-offensive-language' | German | detecting offensive language | [GermEval 2018 Task 1](https://projects.fzai.h-da.de/iggsa/projekt/) |  **75.71** (Macro F1) |




