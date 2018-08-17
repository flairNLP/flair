# Tutorial 5: Training a Model

This part of the tutorial shows how you can train your own sequence labeling and text
classification models using state-of-the-art word embeddings.

## Reading an Evaluation Dataset

Flair provides helper methods to read common NLP datasets, such as the CoNLL-03 and CoNLL-2000 evaluation datasets,
and the CoNLL-U format. These might be interesting to you if you want to train your own sequence labelers.

All helper methods for reading data are bundled in the `NLPTaskDataFetcher` class. One option for you is to follow
the instructions for putting the training data in the appropriate folder structure, and use the prepared functions.
For instance, if you want to use the CoNLL-03 data, get it from the task web site and place train, test and dev data
in `/resources/tasks/conll_03/` as follows:

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
sentences_train = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.train')
sentences_dev = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testa')
sentences_test = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testb')

# return corpus
return TaggedCorpus(sentences_train, sentences_dev, sentences_test)
```

The `TaggedCorpus` contains a bunch of useful helper functions. For instance, you can downsample the data by calling
`downsample()` and passing a ratio. So, if you normally get a corpus like this:

```python
original_corpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
```

then you can downsample the corpus, simply like this:

```python
downsampled_corpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
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
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
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
from flair.trainers import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

# 7. start training
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```

Alternatively, try using a stacked embedding with charLM and glove, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments,
check [here](/resources/docs/EXPERIMENTS.md).

## Training a Text Classification Model

Here is example code for training a text classifier over the AGNews corpus, using  a combination of simple GloVe
embeddings and contextual string embeddings. In this example, we downsample the data to 10% of the original data.

The AGNews corpus can be downloaded [here](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html). As
described in the [basics](/resources/docs/TUTORIAL_BASICS.md) the `NLPTaskDataFetcher` expects the data in the
following format:
```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```
Thus, you need to convert the data into the expected format. After you converted the data you can use the
`NLPTask.AG_NEWS` to read the data, if the data files are located in the following folder structure:
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
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS).downsample(0.1)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),
                   CharLMEmbeddings('news-forward'),
                   CharLMEmbeddings('news-backward')]

# 4. init document embedding by passing list of word embeddings
document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_states=512)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = TextClassifierTrainer(classifier, corpus, label_dict)

# 7. start the trainig
trainer.train('resources/ag_news/results',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```

Once the model is trained you can use it to predict the class of new sentences. Just call the `predict` method of the
model.

```
sentences = model.predict(Sentence('France is the current world cup winner.'))

```
The predict method adds the class labels directly to the sentences. Each label has a name and a confidence value.
```
for sentence in sentences:
    print(sentence.labels)
```

## Next

You can now look into [training your own embeddings](/resources/docs/TUTORIAL_TRAINING_LM_EMBEDDINGS.md).
