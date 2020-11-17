from FlairParamOptimizer import search_strategies, search_spaces, orchestrator
import FlairParamOptimizer.parameter_listings.parameters_for_user_input as param

from flair.datasets import UD_ENGLISH

corpus = UD_ENGLISH().downsample(0.5)

search_space = search_spaces.SequenceTaggerSearchSpace()
search_strategy = search_strategies.GridSearch()

search_space.add_tag_type("pos")

search_space.add_budget(param.BudgetConstraint.TIME_IN_H, 24)
search_space.add_evaluation_metric(param.EvaluationMetric.MICRO_F1_SCORE)
search_space.add_optimization_value(param.OptimizationValue.DEV_SCORE)
search_space.add_max_epochs_per_training_run(50)

search_space.add_parameter(param.SequenceTagger.HIDDEN_SIZE, options=[128, 256, 512])
search_space.add_parameter(param.SequenceTagger.DROPOUT, options=[0, 0.1, 0.2, 0.3])
search_space.add_parameter(param.SequenceTagger.WORD_DROPOUT, options=[0, 0.01, 0.05, 0.1])
search_space.add_parameter(param.SequenceTagger.RNN_LAYERS, options=[2, 3, 4, 5, 6])
search_space.add_parameter(param.SequenceTagger.USE_RNN, options=[True, False])
search_space.add_parameter(param.SequenceTagger.USE_CRF, options=[True, False])
search_space.add_parameter(param.SequenceTagger.REPROJECT_EMBEDDINGS, options=[True, False])
search_space.add_parameter(param.SequenceTagger.WORD_EMBEDDINGS, options=[['glove'],
                                                                          ['en'],
                                                                          ['en', 'glove']])

search_strategy.make_configurations(search_space)

orchestrator = orchestrator.Orchestrator(corpus=corpus,
                                           base_path="resources/evaluation_ud-eng_grid-v2",
                                           search_space=search_space,
                                           search_strategy=search_strategy)

orchestrator.optimize()
