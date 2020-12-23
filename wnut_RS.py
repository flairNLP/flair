from flair.datasets import WNUT_17
from flair.hyperparameter import search_strategies, search_spaces
from flair.hyperparameter import parameters
from flair.hyperparameter import orchestrator
from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

def main():
    corpus = WNUT_17()

    # define search space
    search_space = search_spaces.SequenceTaggerSearchSpace()
    search_space.add_tag_type("ner")

    search_strategy = search_strategies.RandomSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.TIME_IN_H, 24)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(50)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])
    search_space.add_parameter(parameters.ModelTrainer.ANNEAL_FACTOR, options=[0.25, 0.5])
    search_space.add_parameter(parameters.Optimizer.WEIGHT_DECAY, options=[1e-2, 0])

    search_space.add_parameter(parameters.SequenceTagger.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_parameter(parameters.SequenceTagger.DROPOUT, options=[0, 0.1, 0.2, 0.3])
    search_space.add_parameter(parameters.SequenceTagger.WORD_DROPOUT, options=[0, 0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.SequenceTagger.RNN_LAYERS, options=[2, 3, 4, 5, 6])
    search_space.add_parameter(parameters.SequenceTagger.USE_RNN, options=[True, False])
    search_space.add_parameter(parameters.SequenceTagger.USE_CRF, options=[True, False])
    search_space.add_parameter(parameters.SequenceTagger.REPROJECT_EMBEDDINGS, options=[True, False])
    search_space.add_word_embeddings(options=[[WordEmbeddings('glove'), WordEmbeddings('en')],
                                              [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'), WordEmbeddings('glove')],
                                              [TransformerWordEmbeddings('bert-base-uncased')]])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path="evaluation_results/wnut/random_search",
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

if __name__ == "__main__":
    main()