# Tutorial 6: Creating a Corpus

This part of the tutorial shows how you can load your own corpus for training your own model later on.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library.


## Reading A Sequence Labeling Dataset

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

Our text classification data format is based on the 
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

## Downloading A Dataset

Flair also supports a growing list of datasets out of the box. You can simply load your preferred 
dataset by calling a dataset name, such as:

```python
corpus = flair.datasets.UD_ENGLISH()
```

This line of code will download the Universal Dependency Treebank for English and put it into `~/.flair/datasets/ud_english`.
The method returns a `Corpus` which can be directly used to train your model.

The following datasets are supported: 

Chunking: [CONLL_2000](https://www.clips.uantwerpen.be/conll2000/chunking/)

Named Entity Recognition:
[CONLL_03_DUTCH](https://www.clips.uantwerpen.be/conll2002/ner/), 
[CONLL_03_SPANISH](https://www.clips.uantwerpen.be/conll2002/ner/),
[WNUT_17](https://noisy-text.github.io/2017/files/),
[NER_BASQUE](http://ixa2.si.ehu.eus/eiec/),
[WIKINER_ENGLISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner),
[WIKINER_GERMAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner),
[WIKINER_DUTCH](https://github.com/dice-group/FOX/tree/master/input/Wikiner),
[WIKINER_FRENCH](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 
[WIKINER_ITALIAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 
[WIKINER_SPANISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 
[WIKINER_PORTUGUESE](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 
[WIKINER_POLISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 
[WIKINER_RUSSIAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner), 

Universal Dependency Treebanks:
[UD_DUTCH](https://github.com/UniversalDependencies/UD_Dutch-Alpino),
[UD_CROATIAN](https://github.com/UniversalDependencies/UD_Croatian-SET) ,
[UD_FRENCH](https://github.com/UniversalDependencies/UD_French-GSD),
[UD_SERBIAN](https://github.com/UniversalDependencies/UD_Serbian-SET),
[UD_ITALIAN](https://github.com/UniversalDependencies/UD_Italian-ISDT),
[UD_BULGARIAN](https://github.com/UniversalDependencies/UD_Bulgarian-BTB),
[UD_SPANISH](https://github.com/UniversalDependencies/UD_Spanish-GSD),
[UD_ARABIC](https://github.com/UniversalDependencies/UD_Arabic-PADT),
[UD_PORTUGUESE](https://github.com/UniversalDependencies/UD_Portuguese-Bosque),
[UD_HEBREW](https://github.com/UniversalDependencies/UD_Hebrew-HTB),
[UD_ROMANIAN](https://github.com/UniversalDependencies/UD_Romanian-RRT),
[UD_TURKISH](https://github.com/UniversalDependencies/UD_Turkish-IMST),
[UD_CATALAN](https://github.com/UniversalDependencies/UD_Catalan-AnCora), 
[UD_PERSIAN](https://github.com/UniversalDependencies/UD_Persian-Seraji),
[UD_POLISH](https://github.com/UniversalDependencies/UD_Polish-LFG), 
[UD_RUSSIAN](https://github.com/UniversalDependencies/UD_Russian-SynTagRus),
[UD_CZECH](https://github.com/UniversalDependencies/UD_Czech-PDT), 
[UD_HINDI](https://github.com/UniversalDependencies/UD_Hindi-HDTB),
[UD_SLOVAK](https://github.com/UniversalDependencies/UD_Slovak-SNK), 
[UD_INDONESIAN](https://github.com/UniversalDependencies/UD_Indonesian-GSD),
[UD_SWEDISH](https://github.com/UniversalDependencies/UD_Swedish-Talbanken), 
[UD_JAPANESE](https://github.com/UniversalDependencies/UD_Japanese-GSD),
[UD_DANISH](https://github.com/UniversalDependencies/UD_Danish-DDT), 
[UD_CHINESE](https://github.com/UniversalDependencies/UD_Chinese-GSD),
[UD_NORWEGIAN](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal), 
[UD_KOREAN](https://github.com/UniversalDependencies/UD_Korean-Kaist),
[UD_ENGLISH](https://github.com/UniversalDependencies/UD_English-EWT), 
[UD_FINNISH](https://github.com/UniversalDependencies/UD_Finnish-TDT), 
[UD_BASQUE](https://github.com/UniversalDependencies/UD_Basque-BDT),
[UD_GERMAN](https://github.com/UniversalDependencies/UD_German-GSD),
[UD_SLOVENIAN](https://github.com/UniversalDependencies/UD_Slovenian-SSJ),

Text Classification:
[IMDB](http://ai.stanford.edu/~amaas/data/sentiment/), 
[NEWSGROUPS](http://qwone.com/~jason/20Newsgroups/), 
[TREC_6](http://cogcomp.org/Data/QA/QC/), 
[TREC_50](http://cogcomp.org/Data/QA/QC/),

Text Regression:
[WASSA_ANGER](https://competitions.codalab.org/competitions/16380#learn_the_details), 
[WASSA_FEAR](https://competitions.codalab.org/competitions/16380#learn_the_details), 
[WASSA_JOY](https://competitions.codalab.org/competitions/16380#learn_the_details), 
[WASSA_SADNESS](https://competitions.codalab.org/competitions/16380#learn_the_details), 

## The Corpus Object

The `Corpus` represents your entire dataset. A `Corpus` consists of a list of `train` sentences,
a list of `dev` sentences, and a list of `test` sentences.

A `Corpus` contains a bunch of useful helper functions.
For instance, you can downsample the data by calling `downsample()` and passing a ratio. So, if you normally get a 
corpus like this:

```python
original_corpus = flair.datasets.UD_ENGLISH()
```

then you can downsample the corpus, simply like this:

```python
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
```

If you print both corpora, you see that the second one has been downsampled to 10% of the data.

```python
print("--- 1 Original ---")
print(original_corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```

This should print:

```console
--- 1 Original ---
TaggedCorpus: 12543 train + 2002 dev + 2077 test sentences

--- 2 Downsampled ---
TaggedCorpus: 1255 train + 201 dev + 208 test sentences
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
corpus = flair.datasets.NLPTask.TREC_6()
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

## The MultiCorpus Object

If you want to train multiple tasks at once, you can use the `MultiCorpus` object.
To initiate the `MultiCorpus` you first need to create any number of `Corpus` objects. Afterwards, you can pass
a list of `Corpus` to the `MultiCorpus` object.

```text
english_corpus = flair.datasets.UD_ENGLISH()
german_corpus = flair.datasets.UD_GERMAN()
dutch_corpus = flair.datasets.UD_DUTCH()

multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])
```

The `MultiCorpus` inherits from `Corpus`, so you can use it like any other corpus to train your models. 


## Next

You can now look into [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).
