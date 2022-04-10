# Tutorial 2: Tagging your Text

This is part 2 of the tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. Here, we show how to use our pre-trained models to
tag your text.

## Tagging with Pre-Trained Sequence Tagging Models

Let's use a pre-trained model for named entity recognition (NER). 
This model was trained over the English CoNLL-03 task and can recognize 4 different entity
types.

```python
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
```
All you need to do is use the `predict()` method of the tagger on a sentence. This will add predicted tags to the tokens
in the sentence. Lets use a sentence with two named entities:

```python
from flair.data import Sentence

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

Showing us that two entities are labeled in this sentence: "George Washington" as PER (person) and "Washington" 
as LOC (location.)

### Getting Annotated Spans

Many sequence labeling methods annotate spans that consist of multiple words,
such as "George Washington" in our example sentence.
You can directly get such spans in a tagged sentence like this:

```python
for entity in sentence.get_spans('ner'):
    print(entity)
```

This should print:
```console
Span[0:2]: "George Washington" → PER (0.9989)
Span[4:5]: "Washington" → LOC (0.9942)
```

Which indicates that "George Washington" is a person (PER) and "Washington" is
a location (LOC). Each such `Span` has a text, its position in the sentence and `Label` 
with a value and a score (confidence in the prediction). 

Let us iterate over the spans again and access these fields:

```python
# iterate over each entity
for entity in sentence.get_spans('ner'):
    
    # print entity text, start_position and end_position
    print(f'entity.text is: "{entity.text}"')
    print(f'entity.start_position is: "{entity.start_position}"')
    print(f'entity.end_position is: "{entity.end_position}"')
    
    # also print the value and score of its "ner"-label
    print(f'entity "ner"-label value is: "{entity.get_label("ner").value}"')
    print(f'entity "ner"-label score is: "{entity.get_label("ner").score}"\n')
```

This should print:
```console
entity.text is: "George Washington"
entity.start_position is: "0"
entity.end_position is: "17"
entity "ner"-label value is: "PER"
entity "ner"-label score is: "0.998886227607727"

entity.text is: "Washington"
entity.start_position is: "26"
entity.end_position is: "36"
entity "ner"-label value is: "LOC"
entity "ner"-label score is: "0.9942097663879395"
```

### Getting Any Labels

Not all our taggers predict on the `Span`-level: Part-of-Speech taggers for instance predict on the `Token`-level, 
while relation extractors predict on the `Relation`-level. If you don't know what objects are predicted 
by a specific tagger, the best way to access predictions is through the generic `get_labels()` method:

```python
for label in sentence.get_labels('ner'):
    print(label)
```

The printout is identical in this case to `sentence.get_spans('ner')`: 

```console
Span[0:2]: "George Washington" → PER (0.9989)
Span[4:5]: "Washington" → LOC (0.9942)
```

You can access the label's value and score, and even access the data point object to which the label attaches:

```python
# iterate over all 'ner'-labels in the sentence
for label in sentence.get_labels('ner'):
    # print label value and score
    print(f'label.value is: "{label.value}"')
    print(f'label.score is: "{label.score}"')
    # access the data point to which label attaches and print its text
    print(f'the text of label.data_point is: "{label.data_point.text}"\n')
```


### Multi-Tagging 

Sometimes you want to predict several types of annotation at once, for instance NER and part-of-speech (POS) tags. 
For this, you can use our new `MultiTagger` object, like this: 

```python
from flair.models import MultiTagger

# load tagger for POS and NER 
tagger = MultiTagger.load(['pos', 'ner'])

# make example sentence
sentence = Sentence("George Washington went to Washington.")

# predict with both models
tagger.predict(sentence)

print(sentence)
``` 

The sentence now has two types of annotation: POS and NER. Print them using `get_labels()`: 

```python
for label in sentence.get_labels('pos'):
    print(label)

for label in sentence.get_labels('ner'):
    print(label)
``` 

This should print first the POS tags, then the NER spans: 

```console
Token[0]: "George" → NNP (1.0)
Token[1]: "Washington" → NNP (1.0)
Token[2]: "went" → VBD (1.0)
Token[3]: "to" → IN (0.7628)
Token[4]: "Washington" → NNP (1.0)
Token[5]: "." → . (1.0)

Span[0:2]: "George Washington" → PER (0.9989)
Span[4:5]: "Washington" → LOC (0.9942)
``` 

You can alternatively just print all labels of the sentence at the same time, regardless of type: 
```python
for label in sentence.labels:
    print(label)
``` 

### List of Pre-Trained Sequence Tagger Models

You choose which pre-trained model you load by passing the appropriate
string to the `load()` method of the `SequenceTagger` class. 

A full list of our current and community-contributed models can be browsed on the [__model hub__](https://huggingface.co/models?library=flair&sort=downloads). 
At least the following pre-trained models are provided (click on an ID link to get more info
for the model and an online demo):

#### English Models

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[ner](https://huggingface.co/flair/ner-english)' | NER (4-class) |  English | Conll-03  |  **93.03** (F1) |
| '[ner-fast](https://huggingface.co/flair/ner-english-fast)' | NER (4-class)  |  English  |  Conll-03  |  **92.75** (F1) | (fast model)
| '[ner-large](https://huggingface.co/flair/ner-english-large)' | NER (4-class)  |  English  |  Conll-03  |  **94.09** (F1) | (large model)
| 'ner-pooled' | NER (4-class)  |  English |  Conll-03  |  **93.24** (F1) | (memory inefficient)
| '[ner-ontonotes](https://huggingface.co/flair/ner-english-ontonotes)' | NER (18-class) |  English | Ontonotes  |  **89.06** (F1) |
| '[ner-ontonotes-fast](https://huggingface.co/flair/ner-english-ontonotes-fast)' | NER (18-class) |  English | Ontonotes  |  **89.27** (F1) | (fast model)
| '[ner-ontonotes-large](https://huggingface.co/flair/ner-english-ontonotes-large)' | NER (18-class) |  English | Ontonotes  |  **90.93** (F1) | (large model)
| '[chunk](https://huggingface.co/flair/chunk-english)' |  Chunking   |  English | Conll-2000     |  **96.47** (F1) |
| '[chunk-fast](https://huggingface.co/flair/chunk-english-fast)' |   Chunking   |  English | Conll-2000     |  **96.22** (F1) |(fast model)
| '[pos](https://huggingface.co/flair/pos-english)' |  POS-tagging |   English |  Ontonotes     |**98.19** (Accuracy) |
| '[pos-fast](https://huggingface.co/flair/pos-english-fast)' |  POS-tagging |   English |  Ontonotes     |  **98.1** (Accuracy) |(fast model)
| '[upos](https://huggingface.co/flair/upos-english)' |  POS-tagging (universal) | English | Ontonotes     |  **98.6** (Accuracy) |
| '[upos-fast](https://huggingface.co/flair/upos-english-fast)' |  POS-tagging (universal) | English | Ontonotes     |  **98.47** (Accuracy) | (fast model)
| '[frame](https://huggingface.co/flair/frame-english)'  |   Frame Detection |  English | Propbank 3.0     |  **97.54** (F1) |
| '[frame-fast](https://huggingface.co/flair/frame-english-fast)'  |  Frame Detection |  English | Propbank 3.0     |  **97.31** (F1) | (fast model)
| 'negation-speculation'  | Negation / speculation |English |  Bioscope | **80.2** (F1) |


#### Multilingual Models

We distribute new models that are capable of handling text in multiple languages within a singular model. 

The NER models are trained over 4 languages (English, German, Dutch and Spanish) and the PoS models over 12 languages (English, German, French, Italian, Dutch, Polish, Spanish, Swedish, Danish, Norwegian, Finnish and Czech).

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[ner-multi](https://huggingface.co/flair/ner-multi)' | NER (4-class) | Multilingual | Conll-03   |  **89.27**  (average F1) | (4 languages)
| '[ner-multi-fast](https://huggingface.co/flair/ner-multi-fast)' | NER (4-class)|  Multilingual |  Conll-03   |  **87.91**  (average F1) | (4 languages)
| '[pos-multi](https://huggingface.co/flair/upos-multi)' |  POS-tagging   |  Multilingual |  UD Treebanks  |  **96.41** (average acc.) |  (12 languages)
| '[pos-multi-fast](https://huggingface.co/flair/upos-multi-fast)' |  POS-tagging |  Multilingual |  UD Treebanks  |  **92.88** (average acc.) | (12 languages) 

You can pass text in any of these languages to the model. In particular, the NER also kind of works for languages it was not trained on, such as French.

#### Models for Other Languages

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- |------------- | ------------ |
| '[ar-ner](https://huggingface.co/megantosh/flair-arabic-multi-ner)' | NER (4-class) | Arabic | AQMAR & ANERcorp (curated) |  **86.66** (F1) | |
| '[ar-pos](https://huggingface.co/megantosh/flair-arabic-dialects-codeswitch-egy-lev)' | NER (4-class) | Arabic (+dialects)| combination of corpora |  | |
| '[de-ner](https://huggingface.co/flair/ner-german)' | NER (4-class) |  German | Conll-03  |  **87.94** (F1) | |
| '[de-ner-large](https://huggingface.co/flair/ner-german-large)' | NER (4-class) |  German | Conll-03  |  **92,31** (F1) | |
| 'de-ner-germeval' | NER (4-class) | German | Germeval  |  **84.90** (F1) | |
| '[de-ner-legal](https://huggingface.co/flair/ner-german-legal)' | NER (legal text) |  German | [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) dataset  |  **96.35** (F1) | |
| 'de-pos' | POS-tagging | German | UD German - HDT  |  **98.50** (Accuracy) | |
| 'de-pos-tweets' | POS-tagging | German | German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |
| 'de-historic-indirect' | historical indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-direct' | historical direct speech |  German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-reported' | historical reported speech | German |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-free-indirect' | historical free-indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| '[fr-ner](https://huggingface.co/flair/ner-french)' | NER (4-class) | French | [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| '[es-ner-large](https://huggingface.co/flair/ner-spanish-large)' | NER (4-class) | Spanish | CoNLL-03  |  **90,54** (F1) | [mhham](https://github.com/mhham) |
| '[nl-ner](https://huggingface.co/flair/ner-dutch)' | NER (4-class) | Dutch |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **92.58** (F1) |  |
| '[nl-ner-large](https://huggingface.co/flair/ner-dutch-large)' | NER (4-class) | Dutch | Conll-03 |  **95,25** (F1) |  |
| 'nl-ner-rnn' | NER (4-class) | Dutch | [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **90.79** (F1) | |
| '[da-ner](https://huggingface.co/flair/ner-danish)' | NER (4-class) | Danish |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| 'da-pos' | POS-tagging | Danish | [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |
| 'ml-pos' | POS-tagging | Malayalam | 30000 Malayalam sentences  | **83** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'ml-upos' | POS-tagging | Malayalam | 30000 Malayalam sentences | **87** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'pt-pos-clinical' | POS-tagging | Portuguese | [PUCPR](https://github.com/HAILab-PUCPR/portuguese-clinical-pos-tagger) | **92.39** | [LucasFerroHAILab](https://github.com/LucasFerroHAILab) for clinical texts |


### Tagging a German sentence

As indicated in the list above, we also provide pre-trained models for languages other than English. To tag a German sentence, just load the appropriate model:

```python

# load model
tagger = SequenceTagger.load('de-ner')

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
tagger = SequenceTagger.load('ar-ner')

# make Arabic sentence
sentence = Sentence("احب برلين")

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
for entity in sentence.get_labels('ner'):
    print(entity)
```

This should print: 
```console
Span[1:2]: "برلين" → LOC (0.9803)
```

### Tagging Multilingual Text

If you have text in many languages (such as English and German), you can use our new multilingual models: 

```python

# load model
tagger = SequenceTagger.load('pos-multi')

# text with English and German sentences
sentence = Sentence('George Washington went to Washington. Dort kaufte er einen Hut.')

# predict PoS tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print (line breaks added for readability): 
```console
Sentence: "George Washington went to Washington . Dort kaufte er einen Hut ." 

→ ["George"/PROPN, "Washington"/PROPN, "went"/VERB, "to"/ADP, "Washington"/PROPN, "."/PUNCT] 

→ ["Dort"/ADV, "kaufte"/VERB, "er"/PRON, "einen"/DET, "Hut"/NOUN, "."/PUNCT]
```

So, both 'went' and 'kaufte' are identified as VERBs in these sentences.

### Experimental: Semantic Frame Detection

For English, we provide a pre-trained model that detects semantic frames in text, trained using Propbank 3.0 frames.
This provides a sort of word sense disambiguation for frame evoking words, and we are curious what researchers might
do with this. 

Here's an example:

```python
# load model
tagger = SequenceTagger.load('frame')

# make English sentence
sentence = Sentence('George returned to Berlin to return his hat.')

# predict NER tags
tagger.predict(sentence)

# go through tokens and print predicted frame (if one is predicted)
for token in sentence:
    print(token)
```
This should print: 

```console
Token[0]: "George"
Token[1]: "returned" → return.01 (0.9951)
Token[2]: "to"
Token[3]: "Berlin"
Token[4]: "to"
Token[5]: "return" → return.02 (0.6361)
Token[6]: "his"
Token[7]: "hat"
Token[8]: "."
```

As we can see, the frame detector makes a distinction in the sentence between two different meanings of the word 'return'.
'return.01' means returning to a location, while 'return.02' means giving something back.

## Tagging a List of Sentences

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of `Sentence` objects to the `.predict()` method.

For instance, you can use the sentence splitter of segtok to split your text:

```python
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

# example text with many sentences
text = "This is a sentence. This is another sentence. I love Berlin."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = SequenceTagger.load('ner')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

Using the `mini_batch_size` parameter of the `.predict()` method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.


## Tagging with Pre-Trained Text Classification Models

Let's use a pre-trained model for detecting positive or negative comments.
This model was trained over a mix of product and movie review datasets and can recognize positive
and negative sentiment in English text.

```python
from flair.models import TextClassifier

# load tagger
classifier = TextClassifier.load('sentiment')
```

All you need to do is use the `predict()` method of the classifier on a sentence. This will add the predicted label to
the sentence. Lets use a sentence with positive sentiment:

```python
# make example sentence
sentence = Sentence("enormously entertaining for moviegoers of any age.")

# call predict
classifier.predict(sentence)

# check prediction
print(sentence)
```

This should print:
```console
Sentence: "enormously entertaining for moviegoers of any age ." → POSITIVE (0.9976)
```

The label POSITIVE is added to the sentence, indicating that this sentence has positive sentiment.

### List of Pre-Trained Text Classification Models

You choose which pre-trained model you load by passing the appropriate
string to the `load()` method of the `TextClassifier` class. Currently, the following pre-trained models
are provided:

| ID | Language | Task | Training Dataset | Accuracy |
| ------------- | ---- | ------------- |------------- |------------- |
| 'sentiment' | English | detecting positive and negative sentiment (transformer-based) | movie and product reviews |  **98.87** |
| 'sentiment-fast' | English | detecting positive and negative sentiment (RNN-based) | movie and product reviews |  **96.83**|
| 'communicative-functions' | English | detecting function of sentence in research paper (BETA) | scholarly papers |  |
| 'de-offensive-language' | German | detecting offensive language | [GermEval 2018 Task 1](https://projects.fzai.h-da.de/iggsa/projekt/) |  **75.71** (Macro F1) |


## Experimental: Relation Extraction

Relations hold between two entities. For instance, a text like "George was born in Washington" 
names two entities and also expresses that there is a born_in relationship between
both. 

We added an experimental relation extraction model 
trained over a modified version of TACRED: `relations`. 
Use this models together with an entity tagger, like so: 
```python
from flair.data import Sentence
from flair.models import RelationExtractor, SequenceTagger

# 1. make example sentence
sentence = Sentence("George was born in Washington")

# 2. load entity tagger and predict entities
tagger = SequenceTagger.load('ner-fast')
tagger.predict(sentence)

# check which entities have been found in the sentence
entities = sentence.get_labels('ner')
for entity in entities:
    print(entity)

# 3. load relation extractor
extractor: RelationExtractor = RelationExtractor.load('relations')

# predict relations
extractor.predict(sentence)

# check which relations have been found
relations = sentence.get_labels('relation')
for relation in relations:
    print(relation)
```

This should print: 

~~~
Span[0:1]: "George" → PER (0.9971)
Span[4:5]: "Washington" → LOC (0.9847)

Relation[0:1][4:5]: "George -> Washington" → born_in (1.0)
~~~

Indicating that a born_in relationship holds between "George" and "Washington"!

## Tagging new classes without training data

In case you need to label classes that are not included you can also try
our pre-trained zero-shot classifier TARS 
(skip ahead to the [zero-shot tutorial](/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)).
TARS can perform text classification for arbitrary classes. 

## Next 

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) to embed your
text. 
