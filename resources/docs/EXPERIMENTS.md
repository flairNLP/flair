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

This allows the `CONLL_03()` corpus object to read the data into our data structures. Initialize the corpus as follows:

```python
from flair.datasets import CONLL_03
corpus: Corpus = CONLL_03(base_path='resources/tasks')
```

This gives you a `Corpus` object that contains the data. Now, select `ner` as the tag you wish to predict and init the embeddings you wish to use.

#### Best Known Configuration

The full code to get a state-of-the-art model for English NER is as follows:

```python
from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = CONLL_03(base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [

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
              train_with_dev=True,
              max_epochs=150)
```

## CoNLL-03 Named Entity Recognition (German)

#### Current best score with Flair

**88.27** F1-score, averaged over 5 runs.

#### Data
Get the [CoNLL-03 data set for German](https://www.clips.uantwerpen.be/conll2003/ner/)
It contains 4 entity classes. Follows the steps on the task Web site to
get the dataset. Please note that there are two versions of this dataset: the original and a 2006 revision that makes some tags more consistent. We always use the 2006 version in our experiments. Once you've generated the corpus, place train, test and dev data in `resources/tasks/conll_03-ger/` as follows:

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
from flair.datasets import CONLL_03_GERMAN
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = CONLL_03_GERMAN(base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [
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
              train_with_dev=True,
              max_epochs=150)
```


## CoNLL-02 Named Entity Recognition (Dutch)

#### Current best score with Flair

**92.38** F1-score, averaged over 5 runs.

#### Data
Data is included in Flair and will get automatically downloaded when you run the script.

#### Best Known Configuration

```python
from flair.data import Corpus
from flair.datasets import CONLL_03_DUTCH
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus: Corpus = CONLL_03_DUTCH()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embeddings = TransformerWordEmbeddings('wietsedv/bert-base-dutch-cased', allow_long_sentences=True)

# initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              train_with_dev=True,
              max_epochs=150)
```


## WNUT-17 Emerging Entity Detection (English)

#### Current best score with Flair

**49.49** F1-score, averaged over 5 runs.

#### Data
Data is included in Flair and will get automatically downloaded when you run the script.

#### Best Known Configuration
Once you have the data, reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with
FastText word embeddings for twitter and crawls. The full code then is as follows:

```python
from flair.data import Corpus
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = WNUT_17()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [
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
              train_with_dev=True,
              max_epochs=150)
```


## Ontonotes Named Entity Recognition (English)

#### Current best score with Flair

**89.3** F1-score, averaged over 2 runs.

#### Data

The [Ontonotes corpus](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) is one of the best resources
for different types of NLP and contains rich NER annotation.

You can easily use the corpus within flair:
```
from flair.datasets import ONTONOTES

corpus = ONTONOTES()
```

#### Best Known Configuration

To reproduce our experiments exactly like for CoNLL-03, just with a different dataset and with
FastText embeddings (they work better on this dataset):

```python
from flair.data import Corpus
from flair.datasets import ONTONOTES
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = ONTONOTES()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [
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
              train_with_dev=True,
              # it's a big dataset so maybe set embeddings_storage_mode to 'none' (embeddings are not kept in memory)
              embeddings_storage_mode='none')
```



## Penn Treebank Part-of-Speech Tagging (English)

#### Current best score with Flair

**97.85** accuracy, averaged over 5 runs.

#### Data

Get the [Penn treebank](https://catalog.ldc.upenn.edu/ldc99t42) and follow the guidelines
in [Collins (2002)](http://www.cs.columbia.edu/~mcollins/papers/tagperc.pdf) to produce train, dev and test splits.
Convert splits into CoNLLU-U format and place train, test and dev data in `/path/to/penn/` as follows:

```
/path/to/penn/test.conll
/path/to/penn/train.conll
/path/to/penn/valid.conll
```

Then, run the experiments with extvec embeddings and contextual string embeddings. Also, select 'pos' as `tag_type`,
so the algorithm knows that POS tags and not NER are to be predicted from this data.

#### Best Known Configuration

```python
from flair.data import Corpus
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = UniversalDependenciesCorpus(base_path='/path/to/penn')

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [
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

trainer.train('resources/taggers/example-pos',
              train_with_dev=True,
              max_epochs=150)
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
from flair.datasets import CONLL_2000
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = CONLL_2000()

# 2. what tag do we want to predict?
tag_type = 'np'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: list[TokenEmbeddings] = [
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

trainer.train('resources/taggers/example-chunk',
              train_with_dev=True,
              max_epochs=150)
```
