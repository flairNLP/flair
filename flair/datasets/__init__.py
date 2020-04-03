# Expose base classses
from .base import DataLoader
from .base import SentenceDataset
from .base import StringDataset
from .base import MongoDataset

# Expose all sequence labeling datasets
from .sequence_labeling import ColumnCorpus
from .sequence_labeling import ColumnDataset
from .sequence_labeling import CONLL_03
from .sequence_labeling import CONLL_03_GERMAN
from .sequence_labeling import CONLL_03_DUTCH
from .sequence_labeling import CONLL_03_SPANISH
from .sequence_labeling import CONLL_2000
from .sequence_labeling import DANE
from .sequence_labeling import GERMEVAL_14
from .sequence_labeling import WIKINER_ENGLISH
from .sequence_labeling import WIKINER_GERMAN
from .sequence_labeling import WIKINER_DUTCH
from .sequence_labeling import WIKINER_FRENCH
from .sequence_labeling import WIKINER_ITALIAN
from .sequence_labeling import WIKINER_SPANISH
from .sequence_labeling import WIKINER_PORTUGUESE
from .sequence_labeling import WIKINER_POLISH
from .sequence_labeling import WIKINER_RUSSIAN
from .sequence_labeling import WNUT_17

# Expose all document classification datasets
from .document_classification import ClassificationCorpus
from .document_classification import ClassificationDataset
from .document_classification import CSVClassificationCorpus
from .document_classification import CSVClassificationDataset
from .document_classification import IMDB
from .document_classification import NEWSGROUPS
from .document_classification import SENTEVAL_CR
from .document_classification import SENTEVAL_MR
from .document_classification import SENTEVAL_MPQA
from .document_classification import SENTEVAL_SUBJ
from .document_classification import SENTEVAL_SST_BINARY
from .document_classification import SENTEVAL_SST_GRANULAR
from .document_classification import TREC_50
from .document_classification import TREC_6
from .document_classification import WASSA_ANGER
from .document_classification import WASSA_FEAR
from .document_classification import WASSA_JOY
from .document_classification import WASSA_SADNESS

# Expose all treebanks
from .treebanks import UniversalDependenciesCorpus
from .treebanks import UniversalDependenciesDataset
from .treebanks import UD_ENGLISH
from .treebanks import UD_GERMAN
from .treebanks import UD_GERMAN_HDT
from .treebanks import UD_DUTCH
from .treebanks import UD_FRENCH
from .treebanks import UD_ITALIAN
from .treebanks import UD_SPANISH
from .treebanks import UD_PORTUGUESE
from .treebanks import UD_ROMANIAN
from .treebanks import UD_CATALAN
from .treebanks import UD_POLISH
from .treebanks import UD_CZECH
from .treebanks import UD_SLOVAK
from .treebanks import UD_SWEDISH
from .treebanks import UD_DANISH
from .treebanks import UD_NORWEGIAN
from .treebanks import UD_FINNISH
from .treebanks import UD_SLOVENIAN
from .treebanks import UD_CROATIAN
from .treebanks import UD_SERBIAN
from .treebanks import UD_BULGARIAN
from .treebanks import UD_ARABIC
from .treebanks import UD_HEBREW
from .treebanks import UD_TURKISH
from .treebanks import UD_PERSIAN
from .treebanks import UD_RUSSIAN
from .treebanks import UD_HINDI
from .treebanks import UD_INDONESIAN
from .treebanks import UD_JAPANESE
from .treebanks import UD_CHINESE
from .treebanks import UD_KOREAN
from .treebanks import UD_BASQUE

# Expose all text-text datasets
from .text_text import ParallelTextCorpus
from .text_text import ParallelTextDataset
from .text_text import OpusParallelCorpus

# Expose all text-image datasets
from .text_image import FeideggerCorpus
from .text_image import FeideggerDataset
