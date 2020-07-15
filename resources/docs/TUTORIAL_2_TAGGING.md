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
sentence = Sentence('George Washington went to Washington .')

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
{'text': 'George Washington went to Washington .',
    'entities': [
        {'text': 'George Washington', 'start_pos': 0, 'end_pos': 17, 'type': 'PER', 'confidence': 0.999},
        {'text': 'Washington', 'start_pos': 26, 'end_pos': 36, 'type': 'LOC', 'confidence': 0.998}
    ]}
```

### List of Pre-Trained Sequence Tagger Models

You choose which pre-trained model you load by passing the appropriate
string to the `load()` method of the `SequenceTagger` class. Currently, the following pre-trained models
are provided:

#### English Models

| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner' | 4-class Named Entity Recognition |  Conll-03  |  **93.03** (F1) |
| 'ner-ontonotes' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.06** (F1) |
| 'chunk' |  Syntactic Chunking   |  Conll-2000     |  **96.47** (F1) |
| 'pos' |  Part-of-Speech Tagging (fine-grained) |  Ontonotes     |  **98.19** (Accuracy) |
| 'upos' |  Part-of-Speech Tagging (universal) |  Ontonotes     |  **98.6** (Accuracy) |
| 'keyphrase' |  Methods and materials in science papers (BETA) |  Semeval2017   |  **47.3** (F1)  |
| 'frame'  |   Semantic Frame Detection |  Propbank 3.0     |  **97.54** (F1) |
| 'negation-speculation'  |  Negations and speculations in biomedical articles  |  Bioscope  |  **80.2** (F1) |


#### Fast English Models

In case you do not have a GPU available, we also distribute smaller models that run faster on CPU.


| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-fast' | 4-class Named Entity Recognition |  Conll-03  |  **92.75** (F1) |
| 'ner-ontonotes-fast' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.27** (F1) |
| 'chunk-fast' |  Syntactic Chunking   |  Conll-2000     |  **96.22** (F1) |
| 'pos-fast' |  Part-of-Speech Tagging (fine-grained) |  Ontonotes     |  **98.1** (Accuracy) |
| 'upos-fast' |  Part-of-Speech Tagging (universal) |  Ontonotes     |  **98.47** (Accuracy) |
| 'frame-fast'  |   Semantic Frame Detection | Propbank 3.0     |  **97.31** (F1) |


#### Multilingual Models

We distribute new models that are capable of handling text in multiple languages within a singular model. 

The NER models are trained over 4 languages (English, German, Dutch and Spanish) and the PoS models over 12 languages (English, German, French, Italian, Dutch, Polish, Spanish, Swedish, Danish, Norwegian, Finnish and Czech).

| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-multi' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **89.27**  (average F1) |
| 'ner-multi-fast' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **87.91**  (average F1) |
| 'ner-multi-fast-learn' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **88.18**  (average F1) |
| 'pos-multi' |  Part-of-Speech Tagging   |  Universal Dependency Treebank (12 languages)  |  **96.41** (average acc.) |
| 'pos-multi-fast' |  Part-of-Speech Tagging |  Universal Dependency Treebank (12 languages)  |  **92.88** (average acc.) |

You can pass text in any of these languages to the model. In particular, the NER also kind of works for languages it was not trained on, such as French.

The 'ner-multi-fast-learn' model is an experimental model that accumulates entity representations over time. 

#### German Models

We also distribute German models.

| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'de-ner' | 4-class Named Entity Recognition |  Conll-03  |  **87.94** (F1) | |
| 'de-ner-germeval' | 4+4-class Named Entity Recognition |  Germeval  |  **84.90** (F1) | |
| 'de-pos' | Part-of-Speech Tagging |  UD German - HDT  |  **98.50** (Accuracy) | |
| 'de-pos-tweets' | Part-of-Speech Tagging |  German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |
| 'de-historic-indirect' | historical German speech and thought (indirect) |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-direct' | historical German speech and thought (direct) |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-reported' | historical German speech and thought (reported) |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-free-indirect' | historical German speech and thought (de-historic-free-indirect) |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |


#### Models for other Languages

Thanks to our contributors we are also able to distribute a couple of models for other languages.

| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'fr-ner' | Named Entity Recognition |  [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| 'nl-ner' | Named Entity Recognition |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **92.58** (F1) |  |
| 'nl-ner-rnn' | Named Entity Recognition |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **90.79** (F1) | |
| 'da-ner' | Named Entity Recognition |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| 'da-pos' | Named Entity Recognition |  [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |
| 'ml-pos' | Part-of-Speech Tagging (fine-grained) |  30000 Malayalam sentences  | **83** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'ml-upos' | Part-of-Speech Tagging (universal)| 30000 Malayalam sentences | **87** | [sabiqueqb](https://github.com/sabiqueqb) |


### Tagging a German sentence

As indicated in the list above, we also provide pre-trained models for languages other than English. To tag a German sentence, just load the appropriate model:

```python

# load model
tagger = SequenceTagger.load('de-ner')

# make German sentence
sentence = Sentence('George Washington ging nach Washington .')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())
```

This should print: 
```console
George <B-PER> Washington <E-PER> ging nach Washington <S-LOC> .
```

### Tagging Multilingual Text

If you have text in many languages (such as English and German), you can use our new multilingual models: 

```python

# load model
tagger = SequenceTagger.load('pos-multi')

# text with English and German sentences
sentence = Sentence('George Washington went to Washington . Dort kaufte er einen Hut .')

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
sentence_1 = Sentence('George returned to Berlin to return his hat .')
sentence_2 = Sentence('He had a look at different hats .')

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

# your text of many sentences
text = "This is a sentence. This is another sentence. I love Berlin."

# use a library to split into sentences
from segtok.segmenter import split_single

sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(text)]

# predict tags for list of sentences
tagger: SequenceTagger = SequenceTagger.load('ner')
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
# load tagger
classifier = TextClassifier.load('sentiment')

# predict for example sentence
sentence = Sentence("enormously entertaining for moviegoers of any age .")
classifier.predict(sentence)

# check prediction
print(sentence)
```

This should print:
```console
Sentence: "enormously entertaining for moviegoers of any age ."   [− Tokens: 8  − Sentence-Labels: {'class': [POSITIVE (0.9976)]}]
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


## Next 

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) to embed your
text.
