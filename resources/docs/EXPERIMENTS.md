# How to Reproduce Experiments 

Here, we detail the steps you need to take to reproduce the experiments in [Akbik et. al (2018)](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
and how to train your own state-of-the-art sequence labelers. 

**Data.** For each experiment, you need to first get the evaluation dataset. Then execute the code as provided in this 
documentation.

**More info.** Also do check out the [tutorial](/resources/docs/TUTORIAL.md)  to get a better overview of how Flair works.


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
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.CONLL_03)
```

This gives you a `TaggedCorpus` object that contains the data. 

Now, select 'ner' as the tag you wish to predict and init the embeddings you wish to use.
 
The full code to get a state-of-the-art model for English NER is as follows: 

```python
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from typing import List
import torch

# 1. get the corpus
task_data_fetcher: NLPTaskDataFetcher = NLPTaskDataFetcher()
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.CONLL_03)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TextEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove')
    ,
    # contextual string embeddings, forward
    CharLMEmbeddings('news-forward')
    ,
    # contextual string embeddings, backward
    CharLMEmbeddings('news-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.tagging_model import SequenceTaggerLSTM

tagger: SequenceTaggerLSTM = SequenceTaggerLSTM(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                                use_crf=False)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer: TagTrain = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=True)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=True,
              train_with_dev=True, anneal_mode=True)
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
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
task_data_fetcher: NLPTaskDataFetcher = NLPTaskDataFetcher()
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.ONTONER)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TextEmbeddings] = [

    WordEmbeddings('ft-crawl')
    ,
    CharLMEmbeddings('news-forward')
    ,
    CharLMEmbeddings('news-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.tagging_model import SequenceTaggerLSTM

tagger: SequenceTaggerLSTM = SequenceTaggerLSTM(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                                use_crf=False)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer: TagTrain = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=True)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=True,
              train_with_dev=True, anneal_mode=True)

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
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
task_data_fetcher: NLPTaskDataFetcher = NLPTaskDataFetcher()
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.CONLL_03_GERMAN)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TextEmbeddings] = [

    WordEmbeddings('ft-german')
    ,
    CharLMEmbeddings('german-forward')
    ,
    CharLMEmbeddings('german-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.tagging_model import SequenceTaggerLSTM

tagger: SequenceTaggerLSTM = SequenceTaggerLSTM(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                                use_crf=False)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer: TagTrain = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=True)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=True,
              train_with_dev=True, anneal_mode=True)
```


## Germeval Named Entity Recognition (German)

**Data.** The [Germeval data set](https://www.clips.uantwerpen.be/conll2003/ner/) is a more recent and more accessible 
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
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# 1. get the corpus
task_data_fetcher: NLPTaskDataFetcher = NLPTaskDataFetcher()
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.GERMEVAL)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TextEmbeddings] = [

    WordEmbeddings('ft-german')
    ,
    CharLMEmbeddings('german-forward')
    ,
    CharLMEmbeddings('german-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.tagging_model import SequenceTaggerLSTM

tagger: SequenceTaggerLSTM = SequenceTaggerLSTM(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                                use_crf=False)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer: TagTrain = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=True)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=True,
              train_with_dev=True, anneal_mode=True)
```