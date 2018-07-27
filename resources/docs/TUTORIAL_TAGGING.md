# Tutorial 2: Tagging your Text

This is part 2 of the tutorial. It assumes that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this library. Here, we show how to use our pre-trained models to tag your text. 

## Tagging with Pre-Trained Models

Let's use a pre-trained model for named entity recognition (NER). 
This model was trained over the English CoNLL-03 task and can recognize 4 different entity
types.

```python
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
```
All you need to do is use the `predict()` method of the tagger on a sentence. This will add predicted tags to the tokens
in the sentence. Lets use a sentence with two named
entities: 

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

You chose which pre-trained model you load by passing the appropriate 
string to the `load()` method of the `SequenceTagger` class. Currently, the following pre-trained models
are provided (more coming): 
 
| ID | Task | Language| Training Dataset | Accuracy | 
| -------------    | ------------- | ------------- |------------- |------------- |
| 'ner' | 4-class Named Entity Recognition | English | Conll-03  |  **93.18** (F1) |
| 'ner-ontonotes' | 12-class Named Entity Recognition | English | Ontonotes  |  **89.62** (F1) |
| 'chunk' |  Syntactic Chunking   | English | Conll-2000     |  **96.68** (F1) |
| 'pos' |  Part-of-Speech Tagging | English | Ontonotes     |  **98.06** (Accuracy) |
| 'frame'  |   Semantic Frame Detection  (***Experimental***)| English | Propbank 3.0     |  **98.00** (Accuracy) |
| 'de-ner' | 4-class Named Entity Recognition | German | Conll-03  |  **88.29** (F1) |
| 'de-ner-germeval' | 4+4-class Named Entity Recognition | German | Germeval  |  **84.53** (F1) |
| 'de-pos' | Part-of-Speech Tagging | German | Universal Dependency Treebank  |  **94.67** (Accuracy) |


So, if you want to use a `SequenceTagger` that performs PoS tagging, instantiate the tagger as follows:

```python
tagger = SequenceTagger.load('pos')
```

## Tagging a German sentence

As indicated in the list above, we also provide pre-trained models for languages other than English. Currently, we
support German and other languages are forthcoming. To tag a German sentence, just load the appropriate model:

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

## Experimental: Semantic Frame Detection

For English, we now provide a pre-trained model that detects semantic frames in text, trained using Propbank 3.0 frames. 
This provides a sort of word sense disambiguation for frame evoking words, and we are curious what researchers might
do with this. 

Here's an example: 

```python
# load model
tagger = SequenceTagger.load('frame')

# make German sentence
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



## Tagging a List of Sentences

TODO: will be added soon



## Next 

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) to embed your text.
