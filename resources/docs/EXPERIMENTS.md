# Best Configurations per Dataset

Here, we collect the best embedding configurations for each NLP task. If
you achieve better numbers, let us know which exact configuration of Flair
you used and we will add your experiment here!

**Data.** For each experiment, you need to first get the evaluation dataset. Then execute the code as provided in this 
documentation. Also check out the [tutorials](/resources/docs/TUTORIAL_1_BASICS.md) to get a better overview of
how Flair works.


## CoNLL-03 Named Entity Recognition (English)

#### Current best score with Flair

**93.16** F1-score, averaged over 5 runs.

#### Data
The [CoNLL-03 data set for English](https://www.clips.uantwerpen.be/conll2003/ner/) is probably the most
well-known dataset to evaluate NER on. It contains 4 entity classes. Follows the steps on the task Web site to 
get the dataset and place train, test and dev data in `/resources/tasks/conll_03/` as follows:

```
resources/tasks/conll_03/eng.testa
resources/tasks/conll_03/eng.testb
resources/tasks/conll_03/eng.train
```

This allows the `NLPTaskDataFetcher` class to read the data into our data structures. Use the `NLPTask` enum to select 
the dataset, as follows: 

```python
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path='resources/tasks')
```

This gives you a `Corpus` object that contains the data. Now, select `ner` as the tag you wish to predict and init the embeddings you wish to use.

#### Best Known Configuration

The full code to get a state-of-the-art model for English NER is as follows: 

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    PooledFlairEmbeddings('news-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('news-backward', pooling='min'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150)
```

## CoNLL-03 Named Entity Recognition (German)

#### Current best score with Flair

**88.27** F1-score, averaged over 5 runs.

#### Data
Get the [CoNLL-03 data set for German](https://www.clips.uantwerpen.be/conll2003/ner/).
It contains 4 entity classes. Follows the steps on the task Web site to
get the dataset and place train, test and dev data in `resources/tasks/conll_03-ger/` as follows:

```
resources/tasks/conll_03-ger/deu.testa
resources/tasks/conll_03-ger/deu.testb
resources/tasks/conll_03-ger/deu.train
```

#### Best Known Configuration
Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with
FastText word embeddings and German contextual string embeddings. The full code then is as follows:

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03_GERMAN, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    PooledFlairEmbeddings('german-forward'),
    PooledFlairEmbeddings('german-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150)
```


## CoNLL-03 Named Entity Recognition (Dutch)

#### Current best score with Flair

**90.44** F1-score, averaged over 5 runs.

#### Data
Data is included in Flair and will get automatically downloaded when you run the script.

#### Best Known Configuration
Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with
FastText word embeddings and German contextual string embeddings. The full code then is as follows:

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03_DUTCH, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('nl'),
    PooledFlairEmbeddings('dutch-forward', pool='mean'),
    PooledFlairEmbeddings('dutch-backward', pool='mean'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150)
```


## WNUT-17 Emerging Entity Detection (English)

#### Current best score with Flair

**49.49** F1-score, averaged over 5 runs.

#### Data
Data is included in Flair and will get automatically downloaded when you run the script.

#### Best Known Configuration
Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with
FastText word embeddings and German contextual string embeddings. The full code then is as follows:

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03_DUTCH, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('crawl'),
    WordEmbeddings('twitter'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150)
```


## Ontonotes Named Entity Recognition (English)

#### Current best score with Flair

**89.3** F1-score, averaged over 2 runs.

#### Data

The [Ontonotes corpus](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) is one of the best resources
for different types of NLP and contains rich NER annotation. Get the corpus and split it into train, test and dev 
splits using the scripts provided by the [CoNLL-12 shared task](http://conll.cemantix.org/2012/data.html). 

Place train, test and dev data in CoNLL-03 format in  `resources/tasks/onto-ner/` as follows:

```
resources/tasks/onto-ner/eng.testa
resources/tasks/onto-ner/eng.testb
resources/tasks/onto-ner/eng.train
```

#### Best Known Configuration

Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with 
FastText embeddings (they work better on this dataset). The full code then is as follows: 

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.ONTONER, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('crawl'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              # it's a big dataset so maybe set embeddings_in_memory to False
              embeddings_in_memory=False)
```



## Penn Treebank Part-of-Speech Tagging (English)

#### Current best score with Flair

**97.85** accuracy, averaged over 5 runs.

#### Data

Get the [Penn treebank](https://catalog.ldc.upenn.edu/ldc99t42) and follow the guidelines
in [Collins (2002)](http://www.cs.columbia.edu/~mcollins/papers/tagperc.pdf) to produce train, dev and test splits.
Convert splits into CoNLLU-U format and place train, test and dev data in `resources/tasks/penn/` as follows: 

```
resources/tasks/penn/test.conll
resources/tasks/penn/train.conll
resources/tasks/penn/valid.conll
```

Then, run the experiments with extvec embeddings and contextual string embeddings. Also, select 'pos' as `tag_type`, 
so the algorithm knows that POS tags and not NER are to be predicted from this data. 

#### Best Known Configuration

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.PENN, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('extvec'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)
# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150,
              # its a big dataset, so maybe set embeddings_in_memory=False
              embeddings_in_memory=True)
```

## CoNLL-2000 Noun Phrase Chunking (English)

#### Current best score with Flair

**96.72** F1-score, averaged over 5 runs.

#### Data
Data is included in Flair and will get automatically downloaded when you run the script.


#### Best Known Configuration
Run the code with extvec embeddings and our proposed contextual string embeddings. Use 'np' as `tag_type`, 
so the algorithm knows that chunking tags and not NER are to be predicted from this data. 

```python
from flair.data import Corpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_2000)

# 2. what tag do we want to predict?
tag_type = 'np'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('extvec'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              max_epochs=150)
```
