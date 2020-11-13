# Tutorial 11: Hyper-parameter Optimization

* __NOTE: (13.11.2020)__ *At the moment you would need to install flair from master branch to be able to follow this.*

In the following we introduce how to optimize your hyper-parameter configuration for your model.

Flair brings three different approaches (*search_strategies*) how to optimize hyper-parameters, namely:
* Grid Search
* Random Search
* Evolutionary Search

These *search_strategies* take a *search_space* as input, acting as a storage for all parameters you want to optimize
over. In order to provide you maximum flexibility for your optimization (i.e. optimize over different Word- or
Document Embeddings), *search_spaces* are task-specific, currently supporting these types of downstream tasks:
* Sequence labeling
* Task classification

## Setup hyper-parameter configuration
```python
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.hyperparameter import search_strategies, search_spaces

# 1. get the corpus                  
corpus: Corpus = TREC_6()

# 2. define task-specific search space
search_space = search_spaces.TextClassifierSearchSpace()

# 3. define a search_strategy
search_strategy = search_strategies.EvolutionarySearch()
```

You then need to set some mandatory parameters and the parameters you want to optimize over. All *add_** functions of
the search space class take either one (only a value) or two parameters (a key and a value). The latter case is usually
needed if you want to assign a parameter to a certain class, i.e. which WordEmbeddings should DocumentRNNEmbeddings use
and which WordEmbeddings should DocumentPoolEmbeddings use?

Once you set up, you need to make configurations out of your parameter combinations based on your chosen search_strategy.
```python
import flair.hyperparameter.parameters as parameter
from flair.embeddings import WordEmbeddings

# mandatory steering parameters

search_space.add_budget(parameter.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(parameter.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(parameter.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(25)

# task specific parameters
search_space.add_parameter(parameter.ModelTrainer.LEARNING_RATE, options=[0.1, 0.05, 0.01, 3e-5])
search_space.add_parameter(parameter.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_word_embeddings(parameter.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[[WordEmbeddings('glove')]])

search_strategy.make_configurations(search_space)
```
 
After that, the orchestrator takes care of the optimization process.
```python
from flair.hyperparameter import orchestrator

orchestrator = orchestrator.Orchestrator(corpus=corpus,
                                         base_path='hyperopt/trec6',
                                         search_space=search_space,
                                         search_strategy=search_strategy)

orchestrator.optimize()
```

When the budget is used up, you'll be provided with the best configuration and the respective weights of the net.