# Tutorial 7: Model Tuning

This is part 7 of the tutorial, in which we look into how we can improve the quality of our model by selecting
the right set of model and hyper parameters.

## Selecting Hyper Parameters

Flair includes a wrapper for the well-known hyper parameter selection tool [hyperopt](https://github.com/hyperopt/hyperopt).

First you need to load your corpus:
```python
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask

# load your corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS)
```

Second you need to define the search space of parameters.
Therefore, you can use all [parameter expressions](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) defined by hyperopt.

```python
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter

# define your search space
search_space = SearchSpace()
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
```

In the last step you have to create the actual parameter selector. 
Depending on the task you need either to define a `TextClassifierParamSelector` or a `SequenceTaggerParamSelector` and 
start the optimization.
You can define the maximum number of evaluation runs hyperopt should perform (`max_evals`).
A evaluation run performs the specified number of epochs (`max_epochs`). 
To overcome the issue of noisy evaluation scores, we take the evaluation scores (either `dev_score` or `dev_loss`) from 
the last three epochs of the evaluation run and take the average over those as final score, which will be passed to 
hyperopt.
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
