# Tutorial 5: Training a Model

This part of the tutorial shows how you can train your own sequence labeling and text
classification models using state-of-the-art word embeddings.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_WORD_EMBEDDING.md) work.


## Reading A Column-Formatted Dataset

Most sequence labeling datasets in NLP use some sort of column format in which each line is a word and each column is
one level of linguistic annotation. See for instance this sentence:

```console
George N B-PER
Washington N I-PER
went V O
to P O
Washington N B-LOC
```

The first column is the word itself, the second coarse PoS tags, and the third BIO-annotated NER tags. To read such a dataset,
define the column structure as a dictionary and use a helper method.

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from pathlib import Path

# define columns
columns = {0: 'text', 1: 'pos', 2: 'np'}

# this is the folder in which train, test and dev files reside
data_folder = Path('/path/to/data/folder')

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file='train.txt',
                                                              test_file='test.txt',
                                                              dev_file='dev.txt')
```

This gives you a `TaggedCorpus` object that contains the train, dev and test splits, each as a list of `Sentence`.
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

## Downloading A Dataset

Flair also supports a couple of datasets out of the box.
You can simple load your preferred dataset by calling, for example
```python
corpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH)
```
This line of code will download the UD_ENGLISH dataset and puts it into `~/.flair/datasets/ud_english`.
The method returns a `TaggedCorpus` which can be directly used to train your model.

The following datasets are supported:

| `NLPTask` | `NLPTask` | `NLPTask` |
|---|---|---|
| [CONLL_2000](https://www.clips.uantwerpen.be/conll2000/chunking/) | [UD_DUTCH](https://github.com/UniversalDependencies/UD_Dutch-Alpino) | [UD_CROATIAN](https://github.com/UniversalDependencies/UD_Croatian-SET) |
| [CONLL_03_DUTCH](https://www.clips.uantwerpen.be/conll2002/ner/) | [UD_FRENCH](https://github.com/UniversalDependencies/UD_French-GSD) | [UD_SERBIAN](https://github.com/UniversalDependencies/UD_Serbian-SET) |
| [CONLL_03_SPANISH](https://www.clips.uantwerpen.be/conll2002/ner/) | [UD_ITALIAN](https://github.com/UniversalDependencies/UD_Italian-ISDT) | [UD_BULGARIAN](https://github.com/UniversalDependencies/UD_Bulgarian-BTB) |
| [WNUT_17](https://noisy-text.github.io/2017/files/) | [UD_SPANISH](https://github.com/UniversalDependencies/UD_Spanish-GSD) | [UD_ARABIC](https://github.com/UniversalDependencies/UD_Arabic-PADT) |
| [WIKINER_ENGLISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_PORTUGUESE](https://github.com/UniversalDependencies/UD_Portuguese-Bosque) | [UD_HEBREW](https://github.com/UniversalDependencies/UD_Hebrew-HTB) |
| [WIKINER_GERMAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_ROMANIAN](https://github.com/UniversalDependencies/UD_Romanian-RRT) | [UD_TURKISH](https://github.com/UniversalDependencies/UD_Turkish-IMST) |
| [WIKINER_DUTCH](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_CATALAN](https://github.com/UniversalDependencies/UD_Catalan-AnCora) | [UD_PERSIAN](https://github.com/UniversalDependencies/UD_Persian-Seraji) |
| [WIKINER_FRENCH](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_POLISH](https://github.com/UniversalDependencies/UD_Polish-LFG) | [UD_RUSSIAN](https://github.com/UniversalDependencies/UD_Russian-SynTagRus) |
| [WIKINER_ITALIAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_CZECH](https://github.com/UniversalDependencies/UD_Czech-PDT) | [UD_HINDI](https://github.com/UniversalDependencies/UD_Hindi-HDTB) |
| [WIKINER_SPANISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_SLOVAK](https://github.com/UniversalDependencies/UD_Slovak-SNK) | [UD_INDONESIAN](https://github.com/UniversalDependencies/UD_Indonesian-GSD) |
| [WIKINER_PORTUGUESE](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_SWEDISH](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) | [UD_JAPANESE](https://github.com/UniversalDependencies/UD_Japanese-GSD) |
| [WIKINER_POLISH](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_DANISH](https://github.com/UniversalDependencies/UD_Danish-DDT) | [UD_CHINESE](https://github.com/UniversalDependencies/UD_Chinese-GSD) |
| [WIKINER_RUSSIAN](https://github.com/dice-group/FOX/tree/master/input/Wikiner) | [UD_NORWEGIAN](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) | [UD_KOREAN](https://github.com/UniversalDependencies/UD_Korean-Kaist) |
| [UD_ENGLISH](https://github.com/UniversalDependencies/UD_English-EWT) | [UD_FINNISH](https://github.com/UniversalDependencies/UD_Finnish-TDT) |
| [UD_GERMAN](https://github.com/UniversalDependencies/UD_German-GSD) | [UD_SLOVENIAN](https://github.com/UniversalDependencies/UD_Slovenian-SSJ) |


## The TaggedCorpus Object

The `TaggedCorpus` contains a bunch of useful helper functions. For instance, you can downsample the data by calling
`downsample()` and passing a ratio. So, if you normally get a corpus like this:

```python
original_corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03)
```

then you can downsample the corpus, simply like this:

```python
downsampled_corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03).downsample(0.1)
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
TaggedCorpus: 14987 train + 3466 dev + 3684 test sentences

--- 2 Downsampled ---
TaggedCorpus: 1499 train + 347 dev + 369 test sentences
```

## Training a Sequence Labeling Model

Here is example code for a small NER model trained over CoNLL-03 data, using simple GloVe embeddings.
In this example, we downsample the data to 10% of the original data.

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03).downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use contextual string embeddings
    # CharLMEmbeddings('news-forward'),
    # CharLMEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-ner',
              EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

```

Alternatively, try using a stacked embedding with charLM and glove, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments,
check [here](/resources/docs/EXPERIMENTS.md).

## Training a Text Classification Model

Here is example code for training a text classifier over the AGNews corpus, using  a combination of simple GloVe
embeddings and contextual string embeddings. In this example, we downsample the data to 10% of the original data.

The AGNews corpus can be downloaded [here](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

### Preparing the data

We use the [FastText format](https://fasttext.cc/docs/en/supervised-tutorial.html) for text classification data, in which
each line in the file represents a text document. A document can have one or multiple labels that are defined at the beginning of the line starting with the prefix
`__label__`. This looks like this:


```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```

Point the `NLPTaskDataFetcher` to this file to convert each line to a `Sentence` object annotated with the labels. It returns a list of `Sentence`.

```python
from flair.data_fetcher import NLPTaskDataFetcher
from pathlib import Path

# use your own data path
data_folder = Path('path/to/text-classification/formatted/data')

# get training, test and dev data
sentences: List[Sentence] = NLPTaskDataFetcher.load_classification_corpus(data_folder)
```


### Training the classifier

To train a model, you need to create three files in this way: A train, dev and test file. After you converted the data you can use the
`NLPTaskDataFetcher` to read the data, if the data files are located in the following folder structure:
```
/resources/tasks/ag_news/train.txt
/resources/tasks/ag_news/dev.txt
/resources/tasks/ag_news/test.txt
```

Here the example code for training the text classifier.
```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS).downsample(0.1)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),
                   CharLMEmbeddings('news-forward'),
                   CharLMEmbeddings('news-backward')]

# 4. init document embedding by passing list of word embeddings
document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                                     hidden_size=512,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=256,)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/ag_news/results',
              EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/ag_news/results/loss.tsv')
plotter.plot_weights('resources/ag_news/results/weights.txt')
```

Once the model is trained you can use it to predict the class of new sentences. Just call the `predict` method of the
model.

```python
sentences = model.predict(Sentence('France is the current world cup winner.'))

```
The predict method adds the class labels directly to the sentences. Each label has a name and a confidence value.
```python
for sentence in sentences:
    print(sentence.labels)
```


## Plotting Training Curves and Weights

Flair includes a helper method to plot training curves and weights in the neural network.
The `ModelTrainer` automatically generates a `loss.tsv` and a `weights.txt` file in the result folder.

After training, simple point the plotter to these files:

```python
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')
```

This generates PNG plots in the result folder.


## Resuming Training

If you want to stop the training at some point and resume it at a later point, you should train with the parameter
`checkpoint` set to `True`.
This will save the model plus training parameters after every epoch.
Thus, you can load the model plus trainer at any later point and continue the training exactly there where you have
left.

The example code below shows how to train, stop, and continue training of a `SequenceTagger`.
Same can be done for `TextClassifier`.

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03).downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-ner',
              EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              checkpoint=True)

# 8. stop training at any point

# 9. continue trainer at later point
from pathlib import Path

trainer = ModelTrainer.load_from_checkpoint(Path('resources/taggers/example-ner/checkpoint.pt'), 'SequenceTagger', corpus)
trainer.train('resources/taggers/example-ner',
              EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              checkpoint=True)
```


## Scalability: Training on Large Data Sets

The main thing to consider when using `CharLMEmbeddings` (which you should) is that they are
somewhat costly to generate for large training data sets. Depending on your setup, you can
set options to optimize training time. There are three questions to ask:

1. Do you have a GPU?

`CharLMEmbeddings` are generated using Pytorch RNNs and are thus optimized for GPUs. If you have one,
you can set large mini-batch sizes to make use of batching. If not, you may want to use smaller language models.
For English, we package 'fast' variants of our embeddings, loadable like this: `CharLMEmbeddings('news-forward-fast')`.

Regardless, all computed embeddings get materialized to disk upon first computation. This means that if you rerun an
experiment on the same dataset, they will be retrieved from disk instead of re-computed, potentially saving a lot
of time.

2. Do embeddings for the entire dataset fit into memory?

In the best-case scenario, all embeddings for the dataset fit into your regular memory, which greatly increases
training speed. If this is not the case, you must set the flag `embeddings_in_memory=False` in the respective trainer
 (i.e. `ModelTrainer`) to
avoid memory problems. With the flag, embeddings are either (a) recomputed at each epoch or (b)
retrieved from disk (where they are materialized by default). The second option is the default and is typically
much faster.

3. Do you have a fast hard drive?

You benefit the most from the default behavior of storing computed embeddings on disk for later retrieval
if your disk is large and fast. If you either do not have a lot of disk space, or a really slow hard drive,
you should disable this option. You can do this when instantiating the embeddings by setting `use_cache=False`. So
instantiate like this: `CharLMEmbeddings('news-forward-fast', use_cache=False')`



## Next

You can now look into [training your own embeddings](/resources/docs/TUTORIAL_TRAINING_LM_EMBEDDINGS.md).