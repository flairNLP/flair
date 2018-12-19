# Tutorial 7: Training a Model

This part of the tutorial shows how you can train your own sequence labeling and text
classification models using state-of-the-art word embeddings.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work (ideally, you also know how [flair embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md) work). You should also know how to [load
a corpus](/resources/docs/TUTORIAL_6_CORPUS.md).


## Training a Sequence Labeling Model

Here is example code for a small NER model trained over CoNLL-03 data, using simple GloVe embeddings. To run this code, you first need to obtain the CoNLL-03 English dataset (alternatively, use `NLPTaskDataFetcher.load_corpus(NLPTask.WNUT)` instead for a task with freely available data).

In this example, we downsample the data to 10% of the original data:

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

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
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

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

```

Alternatively, try using a stacked embedding with FlairEmbeddings and GloVe, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments,
check [here](/resources/docs/EXPERIMENTS.md).

Once the model is trained you can use it to predict the class of new sentences. Just call the `predict` method of the
model.

```python
# load the model you trained
model = SequenceTagger.load_from_file('resources/taggers/example-ner/final-model.pt')

# create example sentence
sentence = Sentence('I love Berlin')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
```

If the model works well, it will correctly tag 'Berlin' as a location in this example.


## Training a Text Classification Model

Here is example code for training a text classifier over the AGNews corpus, using  a combination of simple GloVe
embeddings and Flair embeddings. You need to download the AGNews first to run this code. 
The AGNews corpus can be downloaded [here](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

In this example, we downsample the data to 10% of the original data.

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS, 'path/to/data/folder').downsample(0.1)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),

                   # comment in flair embeddings for state-of-the-art results 
                   # FlairEmbeddings('news-forward'),
                   # FlairEmbeddings('news-backward'),
                   ]

# 4. init document embedding by passing list of word embeddings
document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                                     hidden_size=512,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=256,
                                                                     )

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/taggers/ag_news',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/ag_news/loss.tsv')
plotter.plot_weights('resources/taggers/ag_news/weights.txt')
```

Once the model is trained you can load it to predict the class of new sentences. Just call the `predict` method of the
model.

```python
classifier = TextClassifier.load_from_file('resources/taggers/ag_news/final-model.pt')

# create example sentence
sentence = Sentence('France is the current world cup winner.')

# predict tags and print
classifier.predict(sentence)

print(sentence.labels)
```


## Multi-Dataset Training

Now, let us train a single model that can PoS tag text in both English and German. To do this, we load both the English and German UD corpora and create a MultiCorpus object. We also use the new multilingual Flair embeddings for this task. 

All the rest is same as before, e.g.: 

```python
from typing import List
from flair.data import MultiCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import FlairEmbeddings, TokenEmbeddings, StackedEmbeddings
from flair.training_utils import EvaluationMetric


# 1. get the corpora - English and German UD
corpus: MultiCorpus = NLPTaskDataFetcher.load_corpora([NLPTask.UD_ENGLISH, NLPTask.UD_GERMAN]).downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'upos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
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

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-universal-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              evaluation_metric=EvaluationMetric.MICRO_ACCURACY)
```

Note that here we use the MICRO_ACCURACY evaluation metric instead of the default MICRO_F1_SCORE. This gives you a multilingual model. Try experimenting with more languages!



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

The main thing to consider when using `FlairEmbeddings` (which you should) is that they are
somewhat costly to generate for large training data sets. Depending on your setup, you can
set options to optimize training time. There are three questions to ask:

1. Do you have a GPU?

`CharLMEmbeddings` are generated using Pytorch RNNs and are thus optimized for GPUs. If you have one,
you can set large mini-batch sizes to make use of batching. If not, you may want to use smaller language models.
For English, we package 'fast' variants of our embeddings, loadable like this: `FlairEmbeddings('news-forward-fast')`.

2. Do embeddings for the entire dataset fit into memory?

In the best-case scenario, all embeddings for the dataset fit into your regular memory, which greatly increases
training speed. If this is not the case, you must set the flag `embeddings_in_memory=False` in the respective trainer
 (i.e. `ModelTrainer`) to
avoid memory problems. With the flag, embeddings are either (a) recomputed at each epoch or (b)
retrieved from disk if you choose to materialize to disk. 

3. Do you have a fast hard drive?

If you have a fast hard drive, consider materializing the embeddings to disk. You can do this my instantiating FlairEmbeddings as follows: `FlairEmbeddings('news-forward-fast', use_cache=True)`. This can help if embeddings do not fit into memory. Also if you do not have a GPU and want to do repeat experiments on the same dataset, this helps because embeddings need only be computed once and will then always be retrieved from disk. 


## Next

You can now either look into [optimizing your model](/resources/docs/TUTORIAL_8_MODEL_OPTIMIZATION.md) or
[training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).
