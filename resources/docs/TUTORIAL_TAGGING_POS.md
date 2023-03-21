# Tutorial 2.4: Tagging Parts of Speech in your Text

This is the **part-of-speech tagging** tutorial. It assumes that you're familiar with the
[base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library. 

This tutorial contains the following parts:
* tagging **universal parts of speech** 
* tagging language-specific parts of speech in English
* tagging language-specific parts of speech in non-English languages
* understanding and accessing annotations (important!)
* tagging a whole text corpus
* list of all models in Flair

Let's get started!

## Tagging Universal Parts-of-Speech (UPOS) 

Universal parts-of-speech are a set of minimal syntactic units that exist across languages. For instance, most languages
will have VERBs or NOUNs. 

To tag upos in **English**, do: 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('upos')

# make a sentence
sentence = Sentence('Dirk went to the store.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[6]: "Dirk went to the store." → ["Dirk"/PROPN, "went"/VERB, "to"/ADP, "the"/DET, "store"/NOUN, "."/PUNCT]
```

This indicates for instance that "went" is a VERB and that "store" is a NOUN.

## Tagging Universal Parts-of-Speech (UPOS) in Multilingual Text

We ship models trained over 14 langages to tag upos in **multilingual text**. Use like this: 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load model
tagger = Classifier.load('pos-multi')

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

However note that they were trained for a mix of European languages and therefore will not work for other languages.

## Tagging Language-Specific Parts-of-Speech (POS) in English

Language-specific parts-of-speech are more fine-grained. For English, we offer several models trained over Ontonotes. 

Use like this:

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('pos')

# make a sentence
sentence = Sentence('Dirk went to the store.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[6]: "Dirk went to the store." → ["Dirk"/NNP, "went"/VBD, "to"/IN, "the"/DT, "store"/NN, "."/.]
```

Look at the tag specification of the Penn Treebank to better understand what these tags mean. 

## Tagging Language-Specific Parts-of-Speech (POS) in Other Languages

We ship with language-specific part-of-speech models for several languages. For instance:

### ... in German 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('de-pos')

# make a sentence
sentence = Sentence('Dort hatte er einen Hut gekauft.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

This should print:
```console
Sentence[7]: "Dort hatte er einen Hut gekauft." → ["Dort"/ADV, "hatte"/VAFIN, "er"/PPER, "einen"/ART, "Hut"/NN, "gekauft"/VVPP, "."/$.]
```


### ... in Ukrainian

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('pos-ukrainian')

# make a sentence
sentence = Sentence("Сьогодні в Знам’янці проживають нащадки поета — родина Шкоди.")

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```


### ... in Arabic

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('ar-pos')

# make a sentence
sentence = Sentence('عمرو عادلي أستاذ للاقتصاد السياسي المساعد في الجامعة الأمريكية  بالقاهرة .')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)
```

## Understanding and Accessing Annotations (important!)

You can access each prediction individually using the `get_labels()` method. Let's use our standard UPOS example to 
tag a sentence: 

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('upos')

# make a sentence
sentence = Sentence('George Washington went to Washington.')

# predict NER tags
tagger.predict(sentence)
```

Use the `get_labels()` method to iterate over all predictions:

```python
for label in sentence.get_labels():
    print(label)
```

This should print each token in the sentence, together with its part-of-speech tag:

```console
Token[0]: "George" → PROPN (0.9998)
Token[1]: "Washington" → PROPN (1.0)
Token[2]: "went" → VERB (1.0)
Token[3]: "to" → ADP (1.0)
Token[4]: "Washington" → PROPN (1.0)
Token[5]: "." → PUNCT (1.0)
```

As you can see, each entity is printed, together with the predicted class. The confidence of the prediction is indicated as a score in brackets.

For each prediction, you can directly access the label value, it's score and the token text:  

```python
# iterate over all labels in the sentence
for label in sentence.get_labels():
    # print label value and score
    print(f'label.value is: "{label.value}"')
    print(f'label.score is: "{label.score}"')
    # access the data point to which label attaches and print its text
    print(f'the text of label.data_point is: "{label.data_point.text}"\n')
```


## Tagging a Whole Text Corpus

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of `Sentence` objects to the `.predict()` method.

For instance, you can use the sentence splitter of segtok to split your text:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "This is a sentence. This is another sentence. I love Berlin."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = Classifier.load('upos')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

Using the `mini_batch_size` parameter of the `.predict()` method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.


## List of POS Models

We end this section with a list of all models we currently ship with Flair. 

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[pos](https://huggingface.co/flair/pos-english)' |  POS-tagging |   English |  Ontonotes     |**98.19** (Accuracy) |
| '[pos-fast](https://huggingface.co/flair/pos-english-fast)' |  POS-tagging |   English |  Ontonotes     |  **98.1** (Accuracy) |(fast model)
| '[upos](https://huggingface.co/flair/upos-english)' |  POS-tagging (universal) | English | Ontonotes     |  **98.6** (Accuracy) |
| '[upos-fast](https://huggingface.co/flair/upos-english-fast)' |  POS-tagging (universal) | English | Ontonotes     |  **98.47** (Accuracy) | (fast model)
| '[pos-multi](https://huggingface.co/flair/upos-multi)' |  POS-tagging   |  Multilingual |  UD Treebanks  |  **96.41** (average acc.) |  (12 languages)
| '[pos-multi-fast](https://huggingface.co/flair/upos-multi-fast)' |  POS-tagging |  Multilingual |  UD Treebanks  |  **92.88** (average acc.) | (12 languages)
| '[ar-pos](https://huggingface.co/megantosh/flair-arabic-dialects-codeswitch-egy-lev)' | POS-tagging | Arabic (+dialects)| combination of corpora |  | |
| 'de-pos' | POS-tagging | German | UD German - HDT  |  **98.50** (Accuracy) | |
| 'de-pos-tweets' | POS-tagging | German | German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |
| 'da-pos' | POS-tagging | Danish | [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |
| 'ml-pos' | POS-tagging | Malayalam | 30000 Malayalam sentences  | **83** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'ml-upos' | POS-tagging | Malayalam | 30000 Malayalam sentences | **87** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'pt-pos-clinical' | POS-tagging | Portuguese | [PUCPR](https://github.com/HAILab-PUCPR/portuguese-clinical-pos-tagger) | **92.39** | [LucasFerroHAILab](https://github.com/LucasFerroHAILab) for clinical texts |
| '[pos-ukrainian](https://huggingface.co/dchaplinsky/flair-uk-pos)' | POS-tagging | Ukrainian |  [Ukrainian UD](https://universaldependencies.org/treebanks/uk_iu/index.html)  | **97.93** (F1)  | [dchaplinsky](https://github.com/dchaplinsky) |

You choose which pre-trained model you load by passing the appropriate string to the `load()` method of the `Classifier` class.

A full list of our current and community-contributed models can be browsed on the [__model hub__](https://huggingface.co/models?library=flair&sort=downloads).

## Next

Go back to the [tagging overview tutorial](/resources/docs/TUTORIAL_TAGGING_OVERVIEW.md) to check out other model types in Flair. Or learn how to train your own NER model in our training tutorial.
