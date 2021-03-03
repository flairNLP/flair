from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings
from flair.hyperparameter import search_spaces, search_strategies, orchestrator, parameters

corpus: Corpus = CONLL_03()
search_space = search_spaces.SequenceTaggerSearchSpace()
search_space.add_tag_type("ner")

search_space.add_budget(parameters.BudgetConstraint.RUNS, 100)
search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)

# Trainingsparameter
search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])

# Modellparameter
search_space.add_parameter(parameters.SequenceTagger.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_word_embeddings(options=[[WordEmbeddings('glove'), WordEmbeddings('en')], [WordEmbeddings('en')]])

search_strategy = search_strategies.EvolutionarySearch(population_size=12,
                                                       cross_rate=0.5,
                                                       mutation_rate=0.1)

# 1) Initialisierung
search_strategy.make_configurations(search_space)

orch = orchestrator.Orchestrator(corpus=corpus,
                                 base_path="storage/folder/",
                                 search_space=search_space,
                                 search_strategy=search_strategy)

orch.optimize()

