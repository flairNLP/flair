from flair.datasets import SENTEVAL_SST_GRANULAR
from flair.hyperparameter import search_strategies, search_spaces
from flair.hyperparameter import parameters
from flair.hyperparameter import orchestrator
from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

def main():
    corpus = SENTEVAL_SST_GRANULAR()

    # define search space
    search_space = search_spaces.TextClassifierSearchSpace(multi_label=True)

    search_strategy = search_strategies.GridSearch()

    # mandatory steering parameters
    search_space.add_budget(parameters.BudgetConstraint.TIME_IN_H, 24)
    search_space.add_evaluation_metric(parameters.EvaluationMetric.MICRO_F1_SCORE)
    search_space.add_optimization_value(parameters.OptimizationValue.DEV_SCORE)
    search_space.add_max_epochs_per_training_run(50)

    search_space.add_parameter(parameters.ModelTrainer.LEARNING_RATE, options=[0.01, 0.05, 0.1])
    search_space.add_parameter(parameters.ModelTrainer.MINI_BATCH_SIZE, options=[16, 32, 64])
    search_space.add_parameter(parameters.ModelTrainer.ANNEAL_FACTOR, options=[0.25, 0.5])
    search_space.add_parameter(parameters.Optimizer.WEIGHT_DECAY, options=[1e-2, 0])

    # Define parameters for document embeddings RNN
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.HIDDEN_SIZE, options=[128, 256, 512])
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.DROPOUT, options=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    search_space.add_parameter(parameters.DocumentRNNEmbeddings.REPROJECT_WORDS, options=[True, False])
    search_space.add_word_embeddings(parameters.DocumentRNNEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove'), WordEmbeddings('en')],
                                              [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'), WordEmbeddings('glove')],
                                              [TransformerWordEmbeddings('bert-base-uncased')]])

    # Define parameters for document embeddings Pool
    search_space.add_word_embeddings(parameters.DocumentPoolEmbeddings.WORD_EMBEDDINGS,
                                     options=[[WordEmbeddings('glove'), WordEmbeddings('en')],
                                              [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'), WordEmbeddings('glove')],
                                              [TransformerWordEmbeddings('bert-base-uncased')]])
    search_space.add_parameter(parameters.DocumentPoolEmbeddings.POOLING, options=['mean', 'max', 'min'])

    # Define parameters for Transformers
    search_space.add_parameter(parameters.TransformerDocumentEmbeddings.MODEL,
                               options=["bert-base-uncased", "distilbert-base-uncased"])
    search_space.add_parameter(parameters.TransformerDocumentEmbeddings.BATCH_SIZE, options=[16, 32, 64])

    search_strategy.make_configurations(search_space)

    orch = orchestrator.Orchestrator(corpus=corpus,
                                     base_path="evaluation_results/sst/grid_search",
                                     search_space=search_space,
                                     search_strategy=search_strategy)

    orch.optimize()

if __name__ == "__main__":
    main()