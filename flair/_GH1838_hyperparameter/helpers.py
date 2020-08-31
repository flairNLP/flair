from .parameters import *
import inspect
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerDocumentEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import SequenceTagger, TextClassifier

BUDGETS = [option.value for option in Budget]

OPTIMIZER_PARAMETERS = [param.value for param in Optimizer]
DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS = inspect.getfullargspec(TransformerDocumentEmbeddings).args
DOCUMENT_RNN_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentRNNEmbeddings).args
DOCUMENT_POOL_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentPoolEmbeddings).args
SEQUENCE_TAGGER_PARAMETERS = inspect.getfullargspec(SequenceTagger).args
TEXT_CLASSIFIER_PARAMETERS = inspect.getfullargspec(TextClassifier).args
TRAINING_PARAMETERS = inspect.getfullargspec(ModelTrainer.train).args + OPTIMIZER_PARAMETERS
MODEL_TRAINER_PARAMETERS = inspect.getfullargspec(ModelTrainer).args