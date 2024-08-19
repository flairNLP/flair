# Tagging parts-of-speech

This tutorials shows you how to do part-of-speech tagging in Flair, showcases univeral and language-specific models, and gives a list of all PoS models in Flair.

## Language-specific parts-of-speech (PoS)


Syntax is fundamentally language-specific, so each language has different fine-grained parts-of-speech. Flair offers models for many languages:  

### ... in English

For English, we offer several models trained over Ontonotes. 

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

This printout tells us for instance that "_Dirk_" is a proper noun (tag: NNP), and "_went_" is a past tense verb (tag: VBD).

```{note}
To better understand what each tag means, consult the [tag specification](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) of the Penn Treebank.
```

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

## Tagging parts-of-speech in any language

Universal parts-of-speech are a set of minimal syntactic units that exist across languages. For instance, most languages
will have VERBs or NOUNs. 


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

You choose which pre-trained model you load by passing the appropriate string to the [`Classifier.load()`](#flair.nn.Classifier.load) method.

A full list of our current and community-contributed models can be browsed on the [__model hub__](https://huggingface.co/models?library=flair&sort=downloads).



