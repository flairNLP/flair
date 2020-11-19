from flair.hyperparameter import search_strategies, search_spaces, orchestrator
import flair.hyperparameter.parameters as param
from flair.datasets import SENTEVAL_SST_GRANULAR
from flair.embeddings import WordEmbeddings
from torch.optim import SGD, Adam

# 1.) Define your corpus
corpus = SENTEVAL_SST_GRANULAR()

# 2.) create an search space
search_space = search_spaces.TextClassifierSearchSpace(multi_label=True)
search_strategy = search_strategies.RandomSearch()

# 3.) depending on your task add the respective parameters you want to optimize over
search_space.add_budget(param.BudgetConstraint.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(15)

#Depending on your downstream task, add embeddings and specify these with the respective Parameters below
search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
search_space.add_parameter(param.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])
search_space.add_parameter(param.ModelTrainer.ANNEAL_FACTOR, options=[0.25, 0.5])
search_space.add_parameter(param.ModelTrainer.OPTIMIZER, options=[SGD, Adam])
search_space.add_parameter(param.Optimizer.WEIGHT_DECAY, options=[1e-2, 0])

#Define parameters for document embeddings RNN
search_space.add_parameter(param.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_parameter(param.DocumentRNNEmbeddings.DROPOUT, options=[0, 0.1, 0.2])
search_space.add_parameter(param.DocumentRNNEmbeddings.REPROJECT_WORDS, options=[True, False])
search_space.add_word_embeddings(param.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[[WordEmbeddings('en'), WordEmbeddings('glove')]])

#Define parameters for document embeddings Pool
search_space.add_word_embeddings(param.DocumentPoolEmbeddings.WORD_EMBEDDINGS, options=[[WordEmbeddings('en'), WordEmbeddings('glove')]])
search_space.add_parameter(param.DocumentPoolEmbeddings.POOLING, options=['mean', 'max', 'min'])

search_space.add_parameter(param.TransformerDocumentEmbeddings.MODEL, options=['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'])
search_space.add_parameter(param.TransformerDocumentEmbeddings.BATCH_SIZE, options=[8, 16, 32])

search_strategy.make_configurations(search_space)

orchestrator = orchestrator.Orchestrator(corpus=corpus,
                                         base_path='resources/evaluation-senteval-sst-random',
                                         search_space=search_space,
                                         search_strategy=search_strategy)

orchestrator.optimize()