# Tutorial 2.6: Other Crazy Models in Flair

This tutorial gives you a tour of **other crazy models** shipped with Flair. These include:
* tagging semantic frames  
* chunking text
* others

Let's get started! 

## Semantic Frame Detection

For English, we provide a pre-trained model that detects semantic frames in text, trained using Propbank 3.0 frames.
This provides a sort of word sense disambiguation for frame evoking words, and we are curious what researchers might
do with this.

Here's an example:

```python
from flair.nn import Classifier
from flair.data import Sentence

# load model
tagger = Classifier.load('frame')

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

As we can see, the frame detector makes a distinction in the sentence between two different meanings of the word 'return'. 'return.01' means returning to a location, while 'return.02' means giving something back.

## Syntactic Chunking

For English, we provide a model for chunking verb and noun phrases, trained using CoNLL 2000. 
from flair.nn import Classifier
from flair.data import Sentence

```python
# load model
tagger = Classifier.load('chunk')

# make English sentence
sentence = Sentence('The quick brown fox jumps over the lazy dog.')

# predict NER tags
tagger.predict(sentence)

# print the chunks
for chunk in sentence.get_labels():
  print(chunk)
```

This should print:

```console
Span[0:4]: "The quick brown fox" → NP (0.9914)
Span[4:5]: "jumps" → VP (1.0)
Span[5:6]: "over" → PP (0.9967)
Span[6:9]: "the lazy dog" → NP (0.9991)
```
This tells us for instance that "the quick brown fox" and "the lazy dog" form syntactic units in this sentence.

### List of Other Models

We end this section with a list of all models we currently ship with Flair:

| ID | Task | Language | Training Dataset | Accuracy | Contributor / Notes |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[chunk](https://huggingface.co/flair/chunk-english)' |  Chunking   |  English | Conll-2000     |  **96.47** (F1) |
| '[chunk-fast](https://huggingface.co/flair/chunk-english-fast)' |   Chunking   |  English | Conll-2000     |  **96.22** (F1) |(fast model)
| '[frame](https://huggingface.co/flair/frame-english)'  |   Frame Detection |  English | Propbank 3.0     |  **97.54** (F1) |
| '[frame-fast](https://huggingface.co/flair/frame-english-fast)'  |  Frame Detection |  English | Propbank 3.0     |  **97.31** (F1) | (fast model)
| 'negation-speculation'  | Negation / speculation |English |  Bioscope | **80.2** (F1) |
| 'communicative-functions' |  detecting function of sentence in research paper (BETA) |  English| scholarly papers |  |
| 'de-historic-indirect' | historical indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-direct' | historical direct speech |  German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-reported' | historical reported speech | German |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-free-indirect' | historical free-indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |


## Next

Go back to the [tagging overview tutorial](/resources/docs/TUTORIAL_TAGGING_OVERVIEW.md) to check out other model types in Flair. Or learn how to train your own NER model in our training tutorial.
