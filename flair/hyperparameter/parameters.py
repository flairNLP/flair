from enum import Enum

"""
Parameter for configuration of hyperparameter optimization
"""
class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"


"""
Parameter for possible optimization values
"""
class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"

"""
Parameters for budget constraint
"""
class BudgetConstraint(Enum):
    RUNS = "runs"
    GENERATIONS = "generations"
    TIME_IN_H = "time_in_h"

"""
Parameter for torch optimizer
"""
class Optimizer(Enum):
    MOMENTUM = "momentum"
    DAMPENING = "dampening"
    WEIGHT_DECAY = "weight_decay"
    NESTEROV = "nesterov"
    AMSGRAD = "amsgrad"
    BETAS = "betas"

"""
Parameter for Model Trainer class und its function train()
"""
class ModelTrainer(Enum):
    OPTIMIZER = "optimizer"
    EPOCH = "epoch"
    USE_TENSORBOARD = "use_tensorboard"
    LEARNING_RATE = "learning_rate"
    MINI_BATCH_SIZE = "mini_batch_size"
    MINI_BATCH_CHUNK_SIZE = "mini_batch_chunk_size"
    MAX_EPOCHS = "max_epochs"
    ANNEAL_FACTOR = "anneal_factor"
    ANNEAL_WITH_RESTARTS = "anneal_with_restarts"
    PATIENCE = "patience"
    INITIAL_EXTRA_PATIENCE = "initial_extra_patience"
    MIN_LEARNING_RATE = "min_learning_rate"
    TRAIN_WITH_DEV = "train_with_dev"
    NUM_WORKERS = "num_workers"

"""
Parameter for Downstream Tasks
"""
class SequenceTagger(Enum):
    HIDDEN_SIZE = "hidden_size"
    WORD_EMBEDDINGS = "embeddings"
    USE_CRF = "use_crf"
    USE_RNN = "use_rnn"
    RNN_LAYERS = "rnn_layers"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    REPROJECT_EMBEDDINGS = "reproject_embeddings"
    TRAIN_INITIAL_HIDDEN_STATE = "train_initial_hidden_state"
    BETA = "beta"

class TextClassifier(Enum):
    BETA = "beta"

"""
Parameter for text classification embeddings
"""
class DocumentRNNEmbeddings(Enum):
    WORD_EMBEDDINGS = "embeddings"
    HIDDEN_SIZE = "hidden_size"
    RNN_LAYERS = "rnn_layers"
    REPROJECT_WORDS = "reproject_words"
    REPROJECT_WORDS_DIMENSION = "reproject_words_dimension"
    BIDIRECTIONAL = "bidirectional"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    RNN_TYPE = "rnn_type"
    FINE_TUNE = "fine_tune"

"""
Parameter for DocumentPool Embeddings
"""
class DocumentPoolEmbeddings(Enum):
    WORD_EMBEDDINGS = "embeddings"
    FINE_TUNE_MODE = "fine_tune_mode"
    POOLING = "pooling"

"""
Parameter for TransformerDocumentEmbeddings
"""
class TransformerDocumentEmbeddings(Enum):
    MODEL = "model"
    FINE_TUNE = "fine_tune"
    BATCH_SIZE = "batch_size"
    LAYERS = "layers"
    USE_SCALER_MIX = "use_scalar_mix"

class DocumentEmbeddings(Enum):
    pass

class DocumentLMEmbeddings(Enum):
    pass

class SentenceTransformerDocumentEmbeddings(Enum):
    pass