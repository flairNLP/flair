# Tutorial 8: Model Tuning

This is part 8 of the tutorial, in which we look into how we can improve the quality of our model by selecting
the right set of model and hyper parameters.

## Selecting Hyper Parameters

Flair includes a wrapper for the well-known hyper parameter selection tool
[hyperopt](https://github.com/hyperopt/hyperopt).

First you need to load your corpus. If you want to load the [AGNews corpus](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
used in the following example, you first need to download it and convert it into the correct format. Please
check [tutorial 6](/resources/docs/TUTORIAL_6_CORPUS.md) for more details.
```python
from flair.datasets import TREC_6

# load your corpus
corpus = TREC_6()
```

Second you need to define the search space of parameters.
Therefore, you can use all
[parameter expressions](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) defined by hyperopt.

```python
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter

# define your search space
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ WordEmbeddings('en') ], 
    [ FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') ]
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
```

Attention: You should always add your embeddings to the search space (as shown above). If you don't want to test
different kind of embeddings, simply pass just one embedding option to the search space, which will then be used in
every test run. Here is an example:
```python
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') ]
])
```

In the last step you have to create the actual parameter selector. 
Depending on the task you need either to define a `TextClassifierParamSelector` or a `SequenceTaggerParamSelector` and 
start the optimization.
You can define the maximum number of evaluation runs hyperopt should perform (`max_evals`).
A evaluation run performs the specified number of epochs (`max_epochs`). 
To overcome the issue of noisy evaluation scores, we take the average over the last three evaluation scores (either
`dev_score` or `dev_loss`) from the evaluation run, which represents the final score and will be passed to hyperopt.
Additionally, you can specify the number of runs per evaluation run (`training_runs`). 
If you specify more than one training run, one evaluation run will be executed the specified number of times.
The final evaluation score will be the average over all those runs.

```python
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue

# create the parameter selector
param_selector = TextClassifierParamSelector(
    corpus, 
    False, 
    'resources/results', 
    'lstm',
    max_epochs=50, 
    training_runs=3,
    optimization_value=OptimizationValue.DEV_SCORE
)

# start the optimization
param_selector.optimize(search_space, max_evals=100)
```

The parameter settings and the evaluation scores will be written to `param_selection.txt` in the result directory.
While selecting the best parameter combination we do not store any model to disk. We also do not perform a test run
during training, we just evaluate the model once after training on the test set for logging purpose.

## Finding the best Learning Rate

The learning rate is one of the most important hyper parameter and it fundamentally depends on the topology of the loss
landscape via the architecture of your model and the training data it consumes. An optimal learning will improve your
training speed and hopefully give more performant models. A simple technique described by Leslie Smith's
[Cyclical Learning Rates for Training](https://arxiv.org/abs/1506.01186) paper is to train your model starting with a
very low learning rate and increases the learning rate exponentially at every batch update of SGD. By plotting the loss
with respect to the learning rate we will typically observe three distinct phases: for low learning rates the loss does
not improve, an optimal learning rate range where the loss drops the steepest and the final phase where the loss
explodes as the learning rate becomes too big. With such a plot, the optimal learning rate selection is as easy as
picking the highest one from the optimal phase.

In order to run such an experiment start with your initialized `ModelTrainer` and call `find_learning_rate()` with the
`base_path` and the file name in which to records the learning rates and losses. Then plot the generated results via the
`Plotter`'s `plot_learning_rate()` function and have a look at the `learning_rate.png` image to select the optimal
learning rate:

```python
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer
from typing import List

# 1. get the corpus
corpus = WNUT_17().downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
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
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. find learning rate
learning_rate_tsv = trainer.find_learning_rate('resources/taggers/example-ner',
                                                    'learning_rate.tsv')

# 8. plot the learning rate finder curve
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_learning_rate(learning_rate_tsv)
```

## Custom Optimizers

You can now use any of PyTorch's optimizers for training when initializing a `ModelTrainer`. To give the optimizer any
extra options just specify it as shown with the `weight_decay` example:

```python
from torch.optim.adam import Adam

trainer = ModelTrainer(tagger, corpus,
                       optimizer=Adam)
                                     
trainer.train(
    "resources/taggers/example",
    weight_decay=1e-4
)
```

## Next

The last tutorial is about [training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).
