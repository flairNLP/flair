import inspect

from .parameters_for_user_input import *

from flair.trainers import ModelTrainer
from flair.embeddings import *
from flair.models import SequenceTagger, TextClassifier
import flair.embeddings.document as document_embeddings
import flair.embeddings.token as token_embeddings


def extractClasses(importModule):
    return [m[0] for m in inspect.getmembers(importModule, inspect.isclass) if m[1].__module__ == importModule.__name__]

DOCUMENT_EMBEDDINGS = extractClasses(document_embeddings)
TOKEN_EMBEDDINGS = extractClasses(token_embeddings)

BUDGETS = [option.value for option in Budget]

OPTIMIZER_PARAMETERS = [param.value for param in Optimizer]
DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS = inspect.getfullargspec(TransformerDocumentEmbeddings).args
DOCUMENT_RNN_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentRNNEmbeddings).args
DOCUMENT_POOL_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentPoolEmbeddings).args
DOCUMENT_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentEmbeddings).args
SENTENCE_TRANSFORMER_DOCUMENT_EMBEDDING_PARAMETERS = inspect.getfullargspec(SentenceTransformerDocumentEmbeddings).args
DOCUMENT_LM_EMBEDDING_PARAMETERS = inspect.getfullargspec(DocumentLMEmbeddings).args
SEQUENCE_TAGGER_PARAMETERS = inspect.getfullargspec(SequenceTagger).args
TEXT_CLASSIFIER_PARAMETERS = inspect.getfullargspec(TextClassifier).args
TRAINING_PARAMETERS = inspect.getfullargspec(ModelTrainer.train).args + OPTIMIZER_PARAMETERS
MODEL_TRAINER_PARAMETERS = inspect.getfullargspec(ModelTrainer).args