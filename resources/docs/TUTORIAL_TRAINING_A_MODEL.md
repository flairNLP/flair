# Tutorial 5: Training a Model

This part of the tutorial shows how you can train your own sequence labeling models using state-of-the-art word 
embeddings. 


## Reading an Evaluation Dataset

Flair provides helper 
methods to read common NLP datasets, such as the CoNLL-03 and CoNLL-2000 evaluation datasets, and the
CoNLL-U format. These might be interesting to you if you want to train your own sequence labelers. 

All helper methods for reading data are bundled in the `NLPTaskDataFetcher` class. One option for you is to follow 
the instructions for putting the training data in the appropriate folder structure, and use the prepared functions. 
For instance, if you want to use the CoNLL-03 data, get it from the task web site and place train, test and dev data in `/resources/tasks/conll_03/` as follows: 

```
/resources/tasks/conll_03/eng.testa
/resources/tasks/conll_03/eng.testb
/resources/tasks/conll_03/eng.train
```

This allows the `NLPTaskDataFetcher` class to read the data into our data structures. Use the `NLPTask` enum to select 
the dataset, as follows: 

```python
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
```

This gives you a `TaggedCorpus` object that contains the data. 

However, this only works if the relative folder structure perfectly matches the presets. If not - or you are using 
a different dataset, you can still use the inbuilt functions to read different CoNLL formats:

```python
# use your own data path
data_folder = 'path/to/your/data'

# get training, test and dev data
sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.train')
sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testa')
sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testb')

# return corpus
return TaggedCorpus(sentences_train, sentences_dev, sentences_test)
```

The `TaggedCorpus` contains a bunch of useful helper functions. For instance, you can downsample the data by calling
`downsample()` and passing a ratio. So, if you normally get a corpus like this:

```python
original_corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
```

then you can downsample the corpus, simply like this: 

```python
downsampled_corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
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


## Training a Model

Here is example code for a small NER model trained over CoNLL-03 data, using simple GloVe embeddings.
In this example, we downsample the data to 10% of the original data. 

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove')

    # comment in this line to use character embeddings
    # , CharacterEmbeddings()

    # comment in these lines to use contextual string embeddings
    # ,
    # CharLMEmbeddings('news-forward')
    # ,
    # CharLMEmbeddings('news-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# initialize trainer
from flair.trainers import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

trainer.train('resources/taggers/example-ner', learning_rate=0.1, mini_batch_size=32, max_epochs=150)
```

Alternatively, try using a stacked embedding with charLM and glove, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments, 
check [here](/resources/docs/EXPERIMENTS.md). 