# Tutorial 6: Loading Training Data

This part of the tutorial shows how you can load a corpus for training a model. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library.


## The Corpus Object

The `Corpus` represents a dataset that you use to train a model. It consists of a list of `train` sentences,
a list of `dev` sentences, and a list of `test` sentences, which correspond to the training, validation and testing 
split during model training.

The following example snippet instantiates the Universal Dependency Treebank for English as a corpus object: 

```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```

The first time you call this snippet, it triggers a download of the Universal Dependency Treebank for English onto your
hard drive. It then reads the train, test and dev splits into the `Corpus` corpus which it returns. Check the length of 
the three splits to see how many Sentences are there:

```python
# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))
```

You can also access the Sentence objects in each split directly. For instance, let us look at the first Sentence in 
the training split of the English UD: 

```python
# print the first Sentence in the training split
print(corpus.test[0])
```
This prints: 
```console
Sentence: "What if Google Morphed Into GoogleOS ?" - 7 Tokens
```

The sentence is fully tagged with syntactic and morphological information. For instance, print the sentence with
PoS tags: 

```python
# print the first Sentence in the training split
print(corpus.test[0].to_tagged_string('pos'))
```

This should print: 

```console
What <WP> if <IN> Google <NNP> Morphed <VBD> Into <IN> GoogleOS <NNP> ? <.>
```

So the corpus is tagged and ready for training.

### Helper functions

A `Corpus` contains a bunch of useful helper functions.
For instance, you can downsample the data by calling `downsample()` and passing a ratio. So, if you normally get a 
corpus like this:

then you can downsample the corpus, simply like this:

```python
import flair.datasets
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
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

For many learning tasks you need to create a target dictionary. Thus, the `Corpus` enables you to create your
tag or label dictionary, depending on the task you want to learn. Simple execute the following code snippet to do so:

```python
# create tag dictionary for a PoS task
corpus = flair.datasets.UD_ENGLISH()
print(corpus.make_tag_dictionary('upos'))

# create tag dictionary for an NER task
corpus = flair.datasets.CONLL_03_DUTCH()
print(corpus.make_tag_dictionary('ner'))

# create label dictionary for a text classification task
corpus = flair.datasets.TREC_6()
print(corpus.make_label_dictionary())
```

Another useful function is `obtain_statistics()` which returns you a python dictionary with useful statistics about your
dataset. Using it, for example, on the IMDB dataset like this

```python
import flair.datasets 
corpus = flair.datasets.TREC_6()
stats = corpus.obtain_statistics()
print(stats)
```
outputs detailed information on the dataset, each split, and the distribution of class labels.

### The MultiCorpus Object

If you want to train multiple tasks at once, you can use the `MultiCorpus` object.
To initiate the `MultiCorpus` you first need to create any number of `Corpus` objects. Afterwards, you can pass
a list of `Corpus` to the `MultiCorpus` object. For instance, the following snippet loads a combination corpus
consisting of the English, German and Dutch Universal Dependency Treebanks.

```python
english_corpus = flair.datasets.UD_ENGLISH()
german_corpus = flair.datasets.UD_GERMAN()
dutch_corpus = flair.datasets.UD_DUTCH()

# make a multi corpus consisting of three UDs
from flair.data import MultiCorpus
multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])
```

The `MultiCorpus` inherits from `Corpus`, so you can use it like any other corpus to train your models. 

## Prepared Datasets

Flair supports a growing list of prepared datasets out of the box. That is, it automatically downloads and sets up the 
data the first time you call the corresponding constructor ID. The following datasets are supported: 


#### Chunking

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'CONLL_2000' | English  |  [CoNLL-2000]((https://www.clips.uantwerpen.be/conll2000/chunking/)) syntactic chunking |


#### Named Entity Recognition

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'CONLL_03_DUTCH' | Dutch  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'CONLL_03_SPANISH' | Spanish  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'WNUT_17' | English  |  [WNUT-17](https://noisy-text.github.io/2017/files/) emerging entity detection |
| 'WIKINER_ENGLISH' | English  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_GERMAN'  | German  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_FRENCH'  | French  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_ITALIAN'  | Italian  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_SPANISH' | Spanish  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_PORTUGUESE' | Portuguese  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_POLISH' | Polish  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_RUSSIAN'  | Russian  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'NER_BASQUE' | Basque  |  [NER dataset for Basque](http://ixa2.si.ehu.eus/eiec/) |


#### Universal Dependency Treebanks

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'UD_ARABIC'| Arabic  |  Universal Dependency Treebank for [Arabic](https://github.com/UniversalDependencies/UD_Arabic-PADT) |
| 'UD_BASQUE'| Basque  |  Universal Dependency Treebank for [Basque](https://github.com/UniversalDependencies/UD_Basque-BDT) |
| 'UD_BULGARIAN'| Bulgarian  |  Universal Dependency Treebank for [Bulgarian](https://github.com/UniversalDependencies/UD_Bulgarian-BTB)
| 'UD_CATALAN', | Catalan  |  Universal Dependency Treebank for [Catalan](https://github.com/UniversalDependencies/UD_Catalan-AnCora) |
| 'UD_CHINESE' | Chinese  |  Universal Dependency Treebank for [Chinese](https://github.com/UniversalDependencies/UD_Chinese-GSD) |
| 'UD_CROATIAN' | Croatian  |  Universal Dependency Treebank for [Croatian](https://github.com/UniversalDependencies/UD_Croatian-SET) |
| 'UD_CZECH' | Czech  |  Very large Universal Dependency Treebank for [Czech](https://github.com/UniversalDependencies/UD_Czech-PDT) |
| 'UD_DANISH' | Danish  |  Universal Dependency Treebank for [Danish](https://github.com/UniversalDependencies/UD_Danish-DDT) |
| 'UD_DUTCH' | Dutch  |  Universal Dependency Treebank for [Dutch](https://github.com/UniversalDependencies/UD_Dutch-Alpino) |
| 'UD_ENGLISH' | English  |  Universal Dependency Treebank for [English](https://github.com/UniversalDependencies/UD_English-EWT) |
| 'UD_FINNISH' | Finnish  |  Universal Dependency Treebank for [Finnish](https://github.com/UniversalDependencies/UD_Finnish-TDT) |
| 'UD_FRENCH' | French  |  Universal Dependency Treebank for [French](https://github.com/UniversalDependencies/UD_French-GSD) |
|'UD_GERMAN' | German  |  Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-GSD) |
|'UD_GERMAN-HDT' | German  |  Very large Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-HDT) |
|'UD_HEBREW' | Hebrew  |  Universal Dependency Treebank for [Hebrew](https://github.com/UniversalDependencies/UD_Hebrew-HTB) |
|'UD_HINDI' | Hindi  |  Universal Dependency Treebank for [Hindi](https://github.com/UniversalDependencies/UD_Hindi-HDTB) |
|'UD_INDONESIAN' | Indonesian  |  Universal Dependency Treebank for [Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD) |
| 'UD_ITALIAN' | Italian  |  Universal Dependency Treebank for [Italian](https://github.com/UniversalDependencies/UD_Italian-ISDT) |
| 'UD_JAPANESE'| Japanese  |  Universal Dependency Treebank for [Japanese](https://github.com/UniversalDependencies/UD_Japanese-GSD) |
|'UD_KOREAN' | Korean  |  Universal Dependency Treebank for [Korean](https://github.com/UniversalDependencies/UD_Korean-Kaist) |
| 'UD_NORWEGIAN',  | Norwegian  |  Universal Dependency Treebank for [Norwegian](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) |
|  'UD_PERSIAN' | Persian / Farsi  |  Universal Dependency Treebank for [Persian](https://github.com/UniversalDependencies/UD_Persian-Seraji) |
| 'UD_POLISH'  |  Polish |  Universal Dependency Treebank for [Polish](https://github.com/UniversalDependencies/UD_Polish-LFG) |
|'UD_PORTUGUESE' | Portuguese  |  Universal Dependency Treebank for [Portuguese](https://github.com/UniversalDependencies/UD_Portuguese-Bosque) |
| 'UD_ROMANIAN' | Romanian  |  Universal Dependency Treebank for [Romanian](https://github.com/UniversalDependencies/UD_Romanian-RRT)  |
| 'UD_RUSSIAN' | Russian  |  Universal Dependency Treebank for [Russian](https://github.com/UniversalDependencies/UD_Russian-SynTagRus) |
| 'UD_SERBIAN' | Serbian  |  Universal Dependency Treebank for [Serbian](https://github.com/UniversalDependencies/UD_Serbian-SET)|
| 'UD_SLOVAK' | Slovak  |  Universal Dependency Treebank for [Slovak](https://github.com/UniversalDependencies/UD_Slovak-SNK) |
| 'UD_SLOVENIAN' | Slovenian  |  Universal Dependency Treebank for [Slovenian](https://github.com/UniversalDependencies/UD_Slovenian-SSJ) |
| 'UD_SPANISH'  | Spanish  |  Universal Dependency Treebank for [Spanish](https://github.com/UniversalDependencies/UD_Spanish-GSD) |
|  'UD_SWEDISH' | Swedish  |  Universal Dependency Treebank for [Swedish](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) |
|  'UD_TURKISH' | Turkish  |  Universal Dependency Treebank for [Tturkish](https://github.com/UniversalDependencies/UD_Turkish-IMST) |



#### Text Classification
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'IMDB' | English |  [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) dataset of movie reviews and sentiment  |
| 'NEWSGROUPS' | English | The popular [20 newsgroups](http://qwone.com/~jason/20Newsgroups/) classification dataset |
| 'TREC_6', 'TREC_50' | English | The [TREC](http://cogcomp.org/Data/QA/QC/) question classification dataset |


#### Text Regression
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'WASSA_ANGER' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (anger) |
| 'WASSA_FEAR' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (fear) |
| 'WASSA_JOY' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (joy) |
| 'WASSA_SADNESS' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (sadness) |


So to load the 20 newsgroups corpus for text classification, simply do:

```python
import flair.datasets
corpus = flair.datasets.NEWSGROUPS()
```

This downloads and sets up everything you need to train your model. 


## Reading Your Own Sequence Labeling Dataset

In cases you want to train over a sequence labeling dataset that is not in the above list, you can load them with the ColumnCorpus object.
Most sequence labeling datasets in NLP use some sort of column format in which each line is a word and each column is
one level of linguistic annotation. See for instance this sentence:

```console
George N B-PER
Washington N I-PER
went V O
to P O
Washington N B-LOC
```

The first column is the word itself, the second coarse PoS tags, and the third BIO-annotated NER tags. To read such a 
dataset, define the column structure as a dictionary and instantiate a `ColumnCorpus`.

```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

```

This gives you a `Corpus` object that contains the train, dev and test splits, each has a list of `Sentence`.
So, to check how many sentences there are in the training split, do

```python
len(corpus.train)
```

You can also access a sentence and check out annotations. Lets assume that the first sentence in the training split is
the example sentence from above, then executing these commands

```python
print(corpus.train[0].to_tagged_string('pos'))
print(corpus.train[0].to_tagged_string('ner'))
```

will print the sentence with different layers of annotation:

```console
George <N> Washington <N> went <V> to <P> Washington <N>

George <B-PER> Washington <I-PER> went to Washington <B-LOC> .
```

## Reading a Text Classification Dataset

If you want to use your own text classification dataset, you must format your data to the  
[FastText format](https://fasttext.cc/docs/en/supervised-tutorial.html), in which each line in the file represents a 
text document. A document can have one or multiple labels that are defined at the beginning of the line starting with 
the prefix `__label__`. This looks like this:

```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```

To create a `Corpus` for a text classification task, you need to have three files (train, dev, and test) in the 
above format located in one folder. This data folder structure could, for example, look like this for the IMDB task:
```text
/resources/tasks/imdb/train.txt
/resources/tasks/imdb/dev.txt
/resources/tasks/imdb/test.txt
```
Now create a `ClassificationCorpus` by pointing to this folder (`/resources/tasks/imdb`). 
Thereby, each line in a file is converted to a `Sentence` object annotated with the labels.

Attention: A text in a line can have multiple sentences. Thus, a `Sentence` object can actually consist of multiple
sentences.

```python
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt')
```

Note that our corpus initializers have methods to automatically look for train, dev and test splits in a folder. So in 
most cases you don't need to specify the file names yourself. Often, this is enough: 

```python
# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus by pointing to folder. Train, dev and test gets identified automatically. 
corpus: Corpus = ClassificationCorpus(data_folder)
```

## Next

You can now look into [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).
