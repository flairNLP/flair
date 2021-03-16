# Expose base classses
from .base import Embeddings
from .base import ScalarMix

# Expose token embedding classes
from .token import TokenEmbeddings
from .token import StackedEmbeddings
from .token import WordEmbeddings
from .token import CharacterEmbeddings
from .token import FlairEmbeddings
from .token import PooledFlairEmbeddings
from .token import TransformerWordEmbeddings
from .token import BPEmbSerializable
from .token import BytePairEmbeddings
from .token import ELMoEmbeddings
from .token import OneHotEmbeddings
from .token import FastTextEmbeddings
from .token import HashEmbeddings
from .token import MuseCrosslingualEmbeddings
from .token import NILCEmbeddings

# Expose document embedding classes
from .document import DocumentEmbeddings
from .document import TransformerDocumentEmbeddings
from .document import DocumentPoolEmbeddings
from .document import DocumentTFIDFEmbeddings
from .document import DocumentRNNEmbeddings
from .document import DocumentLMEmbeddings
from .document import DocumentCNNEmbeddings
from .document import SentenceTransformerDocumentEmbeddings

# Expose image embedding classes
from .image import ImageEmbeddings
from .image import IdentityImageEmbeddings
from .image import PrecomputedImageEmbeddings
from .image import NetworkImageEmbeddings
from .image import ConvTransformNetworkImageEmbeddings

# Expose legacy embedding classes
from .legacy import CharLMEmbeddings
from .legacy import TransformerXLEmbeddings
from .legacy import XLNetEmbeddings
from .legacy import XLMEmbeddings
from .legacy import OpenAIGPTEmbeddings
from .legacy import OpenAIGPT2Embeddings
from .legacy import RoBERTaEmbeddings
from .legacy import CamembertEmbeddings
from .legacy import XLMRobertaEmbeddings
from .legacy import BertEmbeddings
from .legacy import DocumentMeanEmbeddings
from .legacy import DocumentLSTMEmbeddings
from .legacy import ELMoTransformerEmbeddings
