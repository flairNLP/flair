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
sentence = Sentence('George Washington went to Washington.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())
```

This should print: 
```console
George <B-PER> Washington <E-PER> went to Washington <S-LOC> . 
```

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
Span [1,2]: "George Washington"   [− Labels: PER (0.9968)]
Span [5]: "Washington"   [− Labels: LOC (0.9994)]
```

Which indicates that "George Washington" is a person (PER) and "Washington" is
a location (LOC). Each such `Span` has a text, its position in the sentence and `Label` 
with a value and a score (confidence in the prediction). 
You can also get additional information, such as the position offsets of
each entity in the sentence by calling:

```python
print(sentence.to_dict(tag_type='ner'))
```

This should print:
```console
{'text': 'George Washington went to Washington.',
    'entities': [
        {'text': 'George Washington', 'start_pos': 0, 'end_pos': 17, 'type': 'PER', 'confidence': 0.999},
        {'text': 'Washington', 'start_pos': 26, 'end_pos': 36, 'type': 'LOC', 'confidence': 0.998}
    ]}
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

The sentence now has two types of annotation: POS and NER. 

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
print(sentence.to_tagged_string())
```

This should print: 
```console
George <B-PER> Washington <E-PER> ging nach Washington <S-LOC> .
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
LOC [برلين (2)] (0.9803) 
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
print(sentence.to_tagged_string())
```

This should print: 
```console
George <PROPN> Washington <PROPN> went <VERB> to <ADP> Washington <PROPN> . <PUNCT>

Dort <ADV> kaufte <VERB> er <PRON> einen <DET> Hut <NOUN> . <PUNCT>
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
sentence_1 = Sentence('George returned to Berlin to return his hat.')
sentence_2 = Sentence('He had a look at different hats.')

# predict NER tags
tagger.predict(sentence_1)
tagger.predict(sentence_2)

# print sentence with predicted tags
print(sentence_1.to_tagged_string())
print(sentence_2.to_tagged_string())
```
This should print: 

```console
George returned <return.01> to Berlin to return <return.02> his hat .

He had <have.LV> a look <look.01> at different hats .
```

As we can see, the frame detector makes a distinction in sentence 1 between two different meanings of the word 'return'.
'return.01' means returning to a location, while 'return.02' means giving something back.

Similarly, in sentence 2 the frame detector finds a light verb construction in which 'have' is the light verb and
'look' is a frame evoking word.

### Tagging a List of Sentences

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
    print(sentence.to_tagged_string())
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
Sentence: "enormously entertaining for moviegoers of any age."   [− Tokens: 8  − Sentence-Labels: {'class': [POSITIVE (0.9976)]}]
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

## Tagging new classes without training data

In case you need to label classes that are not included you can also try
our pre-trained zero-shot classifier TARS 
(skip ahead to the [zero-shot tutorial](/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)).
TARS can perform text classification for arbitrary classes. 

## Next 

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) to embed your
text. 
