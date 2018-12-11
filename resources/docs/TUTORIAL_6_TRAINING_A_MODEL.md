# Tutorial 6: Training a Model

This part of the tutorial shows how you can train your own sequence labeling and text
classification models using state-of-the-art word embeddings.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work. You should also know how to [load
a corpus](/resources/docs/TUTORIAL_5_CORPUS.md)


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

```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from pathlib import Path


# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS, Path('path/to/data/folder')).downsample(0.1)

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

## Multi-Dataset Training

TODO


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

You can now either look into [optimizing your model](/resources/docs/TUTORIAL_7_MODEL_OPTIMIZATION.md) or
[training your own embeddings](/resources/docs/TUTORIAL_8_TRAINING_LM_EMBEDDINGS.md).