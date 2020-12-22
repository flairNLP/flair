import shutil

import pytest

from flair.embeddings import WordEmbeddings
from flair.hyperparameter import search_spaces, search_strategies, parameters, orchestrator
from flair.datasets import ColumnCorpus, TREC_6

@pytest.mark.integration
def test_sequence_tagger_evolutionary_search(results_base_path, tasks_base_path):
    corpus = ColumnCorpus(
        data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"}
    )

    # define search space
    search_space = search_spaces.SequenceTaggerSearchSpace()
    search_space.add_tag_type("ner")

    search_strategy = search_strategies.EvolutionarySearch(population_size=2)

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.GENERATIONS, 2)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS, options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch


@pytest.mark.integration
def test_sequence_tagger_random_search(results_base_path, tasks_base_path):
    corpus = ColumnCorpus(
        data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"}
    )

    # define search space
    search_space = search_spaces.SequenceTaggerSearchSpace()
    search_space.add_tag_type("ner")

    search_strategy = search_strategies.RandomSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.RUNS, 3)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch


@pytest.mark.integration
def test_sequence_tagger_grid_search(results_base_path, tasks_base_path):
    corpus = ColumnCorpus(
        data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"}
    )

    # define search space
    search_space = search_spaces.SequenceTaggerSearchSpace()
    search_space.add_tag_type("ner")

    search_strategy = search_strategies.GridSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.RUNS, 3)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch


@pytest.mark.integration
def test_text_classifier_evolutionary_search(results_base_path, tasks_base_path):
    corpus = TREC_6()

    # define search space
    search_space = search_spaces.TextClassifierSearchSpace(multi_label=True)

    search_strategy = search_strategies.EvolutionarySearch(population_size=2)

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.GENERATIONS, 2)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch

@pytest.mark.integration
def test_text_classifier_grid_search(results_base_path, tasks_base_path):
    corpus = TREC_6()

    # define search space
    search_space = search_spaces.TextClassifierSearchSpace(multi_label=True)

    search_strategy = search_strategies.GridSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.RUNS, 3)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch

@pytest.mark.integration
def test_text_classifier_random_search(results_base_path, tasks_base_path):
    corpus = TREC_6()

    # define search space
    search_space = search_spaces.TextClassifierSearchSpace(multi_label=True)

    search_strategy = search_strategies.RandomSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.RUNS, 3)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(2)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path=results_base_path,
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

    # clean up results directory
    shutil.rmtree(results_base_path)
    del search_space, search_strategy, orch
