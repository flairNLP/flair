# How to Reproduce Experiments 

Here, we detail the steps you need to take to reproduce the experiments in 
[Akbik et. al (2018)](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
and how to train your own state-of-the-art sequence labelers. 

**Note**: If you want to exactly reproduce the results of the paper, either install version 0.2.1 of Flair from pip, or checkout tag 0.2.0 from the repo. If you just want to train a model, always use the latest version!

**Data.** For each experiment, you need to first get the evaluation dataset. Then execute the code as provided in this 
documentation.

**More info.** Also do check out the [tutorial](/resources/docs/TUTORIAL_1_BASICS.md)  to get a better overview of 
how Flair works.


## CoNLL-03 Named Entity Recognition (English)

**Data.** The [CoNLL-03 data set for English](https://www.clips.uantwerpen.be/conll2003/ner/) is probably the most 
well-known dataset to evaluate NER on. It contains 4 entity classes. Follows the steps on the task Web site to 
get the dataset and place train, test and dev data in `/resources/tasks/conll_03/` as follows: 

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

Now, select `ner` as the tag you wish to predict and init the embeddings you wish to use.
 
The full code to get a state-of-the-art model for English NER is as follows: 

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    CharLMEmbeddings('news-forward'),

    # contextual string embeddings, backward
    CharLMEmbeddings('news-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              embeddings_in_memory=True)
```

## Ontonotes Named Entity Recognition (English)

**Data.** The [Ontonotes corpus](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) is one of the best resources 
for different types of NLP and contains rich NER annotation. Get the corpus and split it into train, test and dev 
splits using the scripts provided by the [CoNLL-12 shared task](http://conll.cemantix.org/2012/data.html). 

Place train, test and dev data in CoNLL-03 format in  `/resources/tasks/onto-ner/` as follows: 

```
/resources/tasks/onto-ner/eng.testa
/resources/tasks/onto-ner/eng.testb
/resources/tasks/onto-ner/eng.train
```

Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with 
FastText embeddings (they work better on this dataset). The full code then is as follows: 


```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.ONTONER)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('crawl'),

    CharLMEmbeddings('news-forward'),

    CharLMEmbeddings('news-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              # it's a big dataset so maybe set embeddings_in_memory to False
              embeddings_in_memory=False)
```

## CoNLL-03 Named Entity Recognition (German)

**Data.** Get the [CoNLL-03 data set for German](https://www.clips.uantwerpen.be/conll2003/ner/). 
It contains 4 entity classes. Follows the steps on the task Web site to 
get the dataset and place train, test and dev data in `/resources/tasks/conll_03-ger/` as follows: 

```
/resources/tasks/conll_03-ger/deu.testa
/resources/tasks/conll_03-ger/deu.testb
/resources/tasks/conll_03-ger/deu.train
```

Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with 
FastText word embeddings and German contextual string embeddings. The full code then is as follows: 

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03_GERMAN)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('de-fasttext'),

    CharLMEmbeddings('german-forward'),

    CharLMEmbeddings('german-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              embeddings_in_memory=True)
```

## Germeval Named Entity Recognition (German)

**Data.** The [Germeval data set](https://sites.google.com/site/germeval2014ner/data) is a more recent and more accessible 
NER data for German. It contains 4 entity classes, plus extra derivative classes. 
Follows the steps on the task Web site to 
get the dataset and place train, test and dev data in `/resources/tasks/germeval/` as follows: 

```
/resources/tasks/germeval/NER-de-dev.tsv
/resources/tasks/germeval/NER-de-test.tsv
/resources/tasks/germeval/NER-de-train.tsv
```

Once you have the data, reproduce our experiments exactly like for the German CoNLL-03: 

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.GERMEVAL)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('de-fasttext'),

    CharLMEmbeddings('german-forward'),

    CharLMEmbeddings('german-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              embeddings_in_memory=True)
```

## Penn Treebank Part-of-Speech Tagging (English)

**Data.** Get the [Penn treebank](https://catalog.ldc.upenn.edu/ldc99t42) and follow the guidelines 
in [Collins (2002)](http://www.cs.columbia.edu/~mcollins/papers/tagperc.pdf) to produce train, dev and test splits.
Convert splits into CoNLLU-U format and place train, test and dev data in `/resources/tasks/penn/` as follows: 

```
/resources/tasks/penn/test.conll
/resources/tasks/penn/train.conll
/resources/tasks/penn/valid.conll
```

Then, run the experiments with extec embeddings and contextual string embeddings. Also, select 'pos' as `tag_type`, 
so the algorithm knows that POS tags and not NER are to be predicted from this data. 


```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.PENN)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('extvec'),

    CharLMEmbeddings('news-forward'),

    CharLMEmbeddings('news-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              # its a bit dataset, so maybe set embeddings_in_memory=False
              embeddings_in_memory=True)
```

## CoNLL-2000 Noun Phrase Chunking (English)

**Data.** Get the [CoNLL-2000 data set for English](https://www.clips.uantwerpen.be/conll2000/chunking/), the most 
well-known dataset to evaluate chunking on. Follows the steps on the task Web site to 
get the dataset and place train and test data in `/resources/tasks/conll_2000/` as follows: 

```
/resources/tasks/conll_2000/test.txt
/resources/tasks/conll_2000/train.txt
```

Our data loader class automatically samples a dev dataset. 

Run the code with extvec embeddings and our proposed contextual string embeddings. Use 'np' as `tag_type`, 
so the algorithm knows that chunking tags and not NER are to be predicted from this data. 

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_2000)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'np'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('extvec'),

    CharLMEmbeddings('news-forward'),

    CharLMEmbeddings('news-backward'),
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

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              embeddings_in_memory=True)
```
