from flair.hyperparameter import search_strategies, search_spaces, orchestrator
import flair.hyperparameter.parameters as param
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings
from torch.optim import SGD, Adam

from flair.datasets import WNUT_17

corpus = WNUT_17()

search_space = search_spaces.SequenceTaggerSearchSpace()
search_strategy = search_strategies.EvolutionarySearch()

search_space.add_tag_type("ner")

search_space.add_budget(param.Budget.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(25)

search_space.add_parameter(param.ModelTrainer.LEARNING_RATE, options=[0.1, 0.05, 0.01, 3e-5])
search_space.add_parameter(param.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])
search_space.add_parameter(param.ModelTrainer.ANNEAL_FACTOR, options=[0.25, 0.5])
search_space.add_parameter(param.ModelTrainer.OPTIMIZER, options=[SGD, Adam])
search_space.add_parameter(param.Optimizer.WEIGHT_DECAY, options=[1e-2, 0])

search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, options=[0, 0.1])
search_space.add_parameter(param.SequenceTagger.WORD_DROPOUT, options=[0, 0.01, 0.05])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, options=[3, 4, 5, 6])
search_space.add_parameter(param.SequenceTagger.USE_RNN, options=[True, False])
search_space.add_word_embeddings(options=[[TransformerWordEmbeddings('bert-base-cased')],
                                          [FlairEmbeddings("news-forward"),
                                           FlairEmbeddings("news-backward"),
                                           WordEmbeddings("glove")],
                                          [TransformerWordEmbeddings('bert-base-cased'),
                                           FlairEmbeddings("news-forward"),
                                           FlairEmbeddings("news-backward")],
                                          [WordEmbeddings("glove"),
                                           WordEmbeddings("en")]])

search_strategy.make_configurations(search_space)

orchestrator = orchestrator.Orchestrator(corpus=corpus,
                                           base_path="resources/evaluation_wnut_grid",
                                           search_space=search_space,
                                           search_strategy=search_strategy)

orchestrator.optimize()
