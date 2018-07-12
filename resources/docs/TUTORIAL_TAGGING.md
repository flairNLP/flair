# Tutorial 2: Tagging your Text

This is part 2 of the tutorial. It assumes that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this library. Here, we show how to use our pre-trained models to tag your text. 

## Tagging with Pre-Trained Models

Let's use a pre-trained model for named entity recognition (NER). 
This model was trained over the English CoNLL-03 task and can recognize 4 different entity
types.

```python
from flair.tagging_model import SequenceTagger

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
George <B-PER> Washington <E-PER> went <O> to <O> Washington <S-LOC> . <O>
```

You chose which pre-trained model you load by passing the appropriate 
string you pass to the `load()` method of the `SequenceTagger` class. Currently, the following pre-trained models
are provided (more coming): 
 
| ID | Task + Training Dataset | Accuracy | 
| -------------    | ------------- | ------------- |
| 'ner' | Conll-03 Named Entity Recognition (English)   |  **93.17** (F1) |
| 'chunk' | Conll-2000 Syntactic Chunking (English)     |  **96.74** (F1) |
| 'pos' | Ontonotes Part-of-Speech Tagging (English)    |  **98.06** (Accuracy) |

So, if you want to use a `SequenceTagger` that performs PoS tagging, instantiate the tagger as follows:

```python
tagger = SequenceTagger.load('pos')
```


## Tagging a List of Sentences

TODO: will be added soon



## Next 

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) to embed your text.
