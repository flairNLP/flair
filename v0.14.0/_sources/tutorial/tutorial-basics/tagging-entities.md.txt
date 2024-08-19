# Tagging entities 

This tutorials shows you how to do named entity recognition, showcases various NER models, and provides a full list of all NER models in Flair.

## Tagging entities with our standard model

Our standard model uses Flair embeddings and was trained over the English CoNLL-03 task and can recognize 4 different entity types. It offers a good tradeoff between accuracy and speed.

As example, let's use the sentence "_George Washington went to Washington._": 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('ner')

# make a sentence
sentence = Sentence('George Washington went to Washington.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence: "George Washington went to Washington ." → ["George Washington"/PER, "Washington"/LOC]
```

The printout tells us that two entities are labeled in this sentence: "George Washington" as PER (person) and "Washington" as LOC (location).

## Tagging entities with our best model

Our best 4-class model is trained using a very large transformer. Use it if accuracy is the most important to you, and speed/memory not so much. 

```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('George Washington went to Washington.')

# load the NER tagger
tagger = Classifier.load('ner-large')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

As you can see, it's the same code, just with '**ner-large**' as model instead of '**ner**'. 
This model also works with most languages. 

```{note}
If you want the fastest model we ship, you can also try 'ner-fast'.
```

## Tagging entities in non-English text

We also have NER models for text in other languages. 

### Tagging a German sentence

To tag a German sentence, just load the appropriate model:

```python

# load model
tagger = Classifier.load('de-ner-large')

# make German sentence
sentence = Sentence('George Washington ging nach Washington.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence: "George Washington ging nach Washington ." → ["George Washington"/PER, "Washington"/LOC]
```

### Tagging an Arabic sentence

Flair also works for languages that write from right to left. To tag an Arabic sentence, just load the appropriate model:

```python

# load model
tagger = Classifier.load('ar-ner')

# make Arabic sentence
sentence = Sentence("احب برلين")

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[2]: "احب برلين" → ["برلين"/LOC]
```

## Tagging Entities with 18 Classes

We also ship models that distinguish between more than just 4 classes. For instance, use our ontonotes models 
to classify 18 different types of entities. 

```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('On September 1st George won 1 dollar while watching Game of Thrones.')

# load the NER tagger
tagger = Classifier.load('ner-ontonotes-large')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

This should print:
```console
Sentence[13]: "On September 1st George won 1 dollar while watching Game of Thrones." → ["September 1st"/DATE, "George"/PERSON, "1 dollar"/MONEY, "Game of Thrones"/WORK_OF_ART]
```

Finding for instance that "Game of Thrones" is a work of art and that "September 1st" is a date.

## Biomedical Data

For biomedical data, we offer the hunflair models that detect 5 different types of biomedical entities. 

```python
from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence('Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome.')

# load the NER tagger
tagger = Classifier.load('bioner')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)
```

This should print:
```console
Sentence[13]: "Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome." → ["Behavioral abnormalities"/Disease, "Fmr1"/Gene, "Mouse"/Species, "Fragile X Syndrome"/Disease]
```

Thus finding entities of classes "Species", "Disease" and "Gene" in this text.

## List of NER Models

We end this section with a list of all models we currently ship with Flair. 

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[ner](https://huggingface.co/flair/ner-english)' | NER (4-class) |  English | Conll-03  |  **93.03** (F1) |
| '[ner-fast](https://huggingface.co/flair/ner-english-fast)' | NER (4-class)  |  English  |  Conll-03  |  **92.75** (F1) | (fast model)
| '[ner-large](https://huggingface.co/flair/ner-english-large)' | NER (4-class)  |  English / Multilingual |  Conll-03  |  **94.09** (F1) | (large model)
| 'ner-pooled' | NER (4-class)  |  English |  Conll-03  |  **93.24** (F1) | (memory inefficient)
| '[ner-ontonotes](https://huggingface.co/flair/ner-english-ontonotes)' | NER (18-class) |  English | Ontonotes  |  **89.06** (F1) |
| '[ner-ontonotes-fast](https://huggingface.co/flair/ner-english-ontonotes-fast)' | NER (18-class) |  English | Ontonotes  |  **89.27** (F1) | (fast model)
| '[ner-ontonotes-large](https://huggingface.co/flair/ner-english-ontonotes-large)' | NER (18-class) |  English / Multilingual | Ontonotes  |  **90.93** (F1) | (large model)
| '[ar-ner](https://huggingface.co/megantosh/flair-arabic-multi-ner)' | NER (4-class) | Arabic | AQMAR & ANERcorp (curated) |  **86.66** (F1) | |
| '[da-ner](https://huggingface.co/flair/ner-danish)' | NER (4-class) | Danish |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| '[de-ner](https://huggingface.co/flair/ner-german)' | NER (4-class) |  German | Conll-03  |  **87.94** (F1) | |
| '[de-ner-large](https://huggingface.co/flair/ner-german-large)' | NER (4-class) |  German / Multilingual | Conll-03  |  **92.31** (F1) | |
| 'de-ner-germeval' | NER (4-class) | German | Germeval  |  **84.90** (F1) | |
| '[de-ner-legal](https://huggingface.co/flair/ner-german-legal)' | NER (legal text) |  German | [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) dataset  |  **96.35** (F1) | |
| '[fr-ner](https://huggingface.co/flair/ner-french)' | NER (4-class) | French | [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| '[es-ner-large](https://huggingface.co/flair/ner-spanish-large)' | NER (4-class) | Spanish | CoNLL-03  |  **90.54** (F1) | [mhham](https://github.com/mhham) |
| '[nl-ner](https://huggingface.co/flair/ner-dutch)' | NER (4-class) | Dutch |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **92.58** (F1) |  |
| '[nl-ner-large](https://huggingface.co/flair/ner-dutch-large)' | NER (4-class) | Dutch | Conll-03 |  **95.25** (F1) |  |
| 'nl-ner-rnn' | NER (4-class) | Dutch | [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **90.79** (F1) | |
| '[ner-ukrainian](https://huggingface.co/dchaplinsky/flair-uk-ner)' | NER (4-class) | Ukrainian |  [NER-UK dataset](https://github.com/lang-uk/ner-uk)  | **86.05** (F1)  | [dchaplinsky](https://github.com/dchaplinsky) |


You choose which pre-trained model you load by passing the appropriate string to the [`Classifier.load()`](#flair.nn.Classifier.load) method.

A full list of our current and community-contributed models can be browsed on the [__model hub__](https://huggingface.co/models?library=flair&sort=downloads).

