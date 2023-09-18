# How to load a prepared dataset

This part of the tutorial shows how you can load a corpus for training a model. 

## The Corpus Object

The [`Corpus`](#flair.data.Corpus) represents a dataset that you use to train a model. It consists of a list of `train` sentences,
a list of `dev` sentences, and a list of `test` sentences, which correspond to the training, validation and testing
split during model training.

The following example snippet instantiates the Universal Dependency Treebank for English as a corpus object:

```python
from flair.datasets import UD_ENGLISH
corpus = UD_ENGLISH()
```

The first time you call this snippet, it triggers a download of the Universal Dependency Treebank for English onto your
hard drive. It then reads the train, test and dev splits into the [`Corpus`](#flair.data.Corpus) which it returns. Check the length of
the three splits to see how many Sentences are there:

```python
# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))
```

You can also access the [`Sentence`](#flair.data.Sentence) objects in each split directly. For instance, let us look at the first Sentence in
the training split of the English UD:

```python
# get the first Sentence in the training split
sentence = corpus.test[0]

# print with all annotations
print(sentence)

# print only with POS annotations (better readability)
print(sentence.to_tagged_string('pos'))
```

The sentence is fully tagged with syntactic and morphological information. With the latter line,
you print out only the POS tags:

```console
Sentence: "What if Google Morphed Into GoogleOS ?" → ["What"/WP, "if"/IN, "Google"/NNP, "Morphed"/VBD, "Into"/IN, "GoogleOS"/NNP, "?"/.]
```

So the corpus is tagged and ready for training.

### Helper functions

A [`Corpus`](#flair.data.Corpus) contains a bunch of useful helper functions.
For instance, you can downsample the data by calling [`Corpus.downsample()`](#flair.data.Corpus.downsample) and passing a ratio. So, if you normally get a
corpus like this:

```python
from flair.datasets import UD_ENGLISH
corpus = UD_ENGLISH()
```

then you can downsample the corpus, simply like this:

```python
from flair.datasets import UD_ENGLISH
downsampled_corpus = UD_ENGLISH().downsample(0.1)
```

If you print both corpora, you see that the second one has been downsampled to 10% of the data.

```python
print("--- 1 Original ---")
print(corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```

This should print:

```console
--- 1 Original ---
Corpus: 12543 train + 2002 dev + 2077 test sentences

--- 2 Downsampled ---
Corpus: 1255 train + 201 dev + 208 test sentences
```

### Creating label dictionaries

For many learning tasks you need to create a "dictionary" that contains all the labels you want to predict.
You can generate this dictionary directly out of the [`Corpus`](#flair.data.Corpus) by calling the method [`Corpus.make_label_dictionary`](#flair.data.Corpus.make_label_dictionary)
and passing the desired `label_type`.

For instance, the UD_ENGLISH corpus instantiated above has multiple layers of annotation like regular
POS tags ('pos'), universal POS tags ('upos'), morphological tags ('tense', 'number'..) and so on.
Create label dictionaries for universal POS tags by passing `label_type='upos'` like this:

```python
# create label dictionary for a Universal Part-of-Speech tagging task
upos_dictionary = corpus.make_label_dictionary(label_type='upos')

# print dictionary
print(upos_dictionary)
```

This will print out the created dictionary:

```console
Dictionary with 17 tags: PROPN, PUNCT, ADJ, NOUN, VERB, DET, ADP, AUX, PRON, PART, SCONJ, NUM, ADV, CCONJ, X, INTJ, SYM
```

#### Dictionaries for other label types

If you don't know the label types in a corpus, just call [`Corpus.make_label_dictionary`](#flair.data.Corpus.make_label_dictionary) with
any random label name (e.g. `corpus.make_label_dictionary(label_type='abcd')`). This will print
out statistics on all label types in the corpus:

```console
The corpus contains the following label types: 'lemma' (in 12543 sentences), 'upos' (in 12543 sentences), 'pos' (in 12543 sentences), 'dependency' (in 12543 sentences), 'number' (in 12036 sentences), 'verbform' (in 10122 sentences), 'prontype' (in 9744 sentences), 'person' (in 9381 sentences), 'mood' (in 8911 sentences), 'tense' (in 8747 sentences), 'degree' (in 7148 sentences), 'definite' (in 6851 sentences), 'case' (in 6486 sentences), 'gender' (in 2824 sentences), 'numtype' (in 2771 sentences), 'poss' (in 2516 sentences), 'voice' (in 1085 sentences), 'typo' (in 399 sentences), 'extpos' (in 185 sentences), 'abbr' (in 168 sentences), 'reflex' (in 98 sentences), 'style' (in 31 sentences), 'foreign' (in 5 sentences)
```

This means that you can create dictionaries for any of these label types for the [`UD_ENGLISH`](#flair.datasets.treebanks.UD_ENGLISH) corpus. Let's create dictionaries for regular part of speech tags
and a morphological number tagging task:

```python
# create label dictionary for a regular POS tagging task
pos_dictionary = corpus.make_label_dictionary(label_type='pos')

# create label dictionary for a morphological number tagging task
tense_dictionary = corpus.make_label_dictionary(label_type='number')
```

If you print these dictionaries, you will find that the POS dictionary contains 50 tags and the number dictionary only 2 for this corpus (singular and plural).


#### Dictionaries for other corpora types

The method [`Corpus.make_label_dictionary`](#flair.data.Corpus.make_label_dictionary) can be used for any corpus, including text classification corpora:

```python
# create label dictionary for a text classification task
from flair.datasets import TREC_6
corpus = TREC_6()
corpus.make_label_dictionary('question_class')
```

### The MultiCorpus Object

If you want to train multiple tasks at once, you can use the [`MultiCorpus`](#flair.data.MultiCorpus) object.
To initiate the [`MultiCorpus`](#flair.data.MultiCorpus) you first need to create any number of [`Corpus`](#flair.data.Corpus) objects. Afterwards, you can pass
a list of [`Corpus`](#flair.data.Corpus) to the [`MultiCorpus`](#flair.data.MultiCorpus) object. For instance, the following snippet loads a combination corpus
consisting of the English, German and Dutch Universal Dependency Treebanks.

```python
from flair.datasets import UD_ENGLISH, UD_GERMAN, UD_DUTCH
english_corpus = UD_ENGLISH()
german_corpus = UD_GERMAN()
dutch_corpus = UD_DUTCH()

# make a multi corpus consisting of three UDs
from flair.data import MultiCorpus
multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])
```

The [`MultiCorpus`](#flair.data.MultiCorpus) inherits from `[`Corpus`](#flair.data.Corpus), so you can use it like any other corpus to train your models.

## Datasets included in Flair

Flair supports many datasets out of the box. It usually automatically downloads and sets up the data the first time you
call the corresponding constructor ID.
The datasets are split into multiple modules, however they all can be imported from `flair.datasets` too.
You can look up the respective modules to find the possible datasets.

The following datasets are supported:

| Task                                | Module                                                                                                                                      |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Named Entity Recognition            | [flair.datasets.sequence_labeling](#flair.datasets.sequence_labeling)                                                                       |
| Text Classification                 | [flair.datasets.document_classification](#flair.datasets.document_classification)                                                           |
| Text Regression                     | [flair.datasets.document_classification](#flair.datasets.document_classification)                                                           |
| Biomedical Named Entity Recognition | [flair.datasets.biomedical](#flair.datasets.biomedical)                                                                                     |
| Entity Linking                      | [flair.datasets.entity_linking](#flair.datasets.entity_linking)                                                                             |
| Relation Extraction                 | [flair.datasets.relation_extraction](#flair.datasets.relation_extraction)                                                                   |
| Sequence Labeling                   | [flair.datasets.sequence_labeling](#flair.datasets.sequence_labeling)                                                                       |
| Glue Benchmark                      | [flair.datasets.text_text](#flair.datasets.text_text) and [flair.datasets.document_classification](#flair.datasets.document_classification) |
| Universal Proposition Banks         | [flair.datasets.treebanks](#flair.datasets.treebanks)                                                                                       |
| Universal Dependency Treebanks      | [flair.datasets.treebanks](#flair.datasets.treebanks)                                                                                       |
| OCR-Layout-NER                      | [flair.datasets.ocr](#flair.datasets.ocr)                                                                                                   |

