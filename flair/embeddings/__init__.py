# Expose base classses
from flair.embeddings.transformer import (
    TransformerEmbeddings,
    TransformerJitDocumentEmbeddings,
    TransformerJitWordEmbeddings,
    TransformerOnnxDocumentEmbeddings,
    TransformerOnnxWordEmbeddings,
)

from .base import Embeddings, ScalarMix

# Expose document embedding classes
from .document import (
    DocumentCNNEmbeddings,
    DocumentEmbeddings,
    DocumentLMEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    DocumentTFIDFEmbeddings,
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings,
)

# Expose image embedding classes
from .image import (
    ConvTransformNetworkImageEmbeddings,
    IdentityImageEmbeddings,
    ImageEmbeddings,
    NetworkImageEmbeddings,
    PrecomputedImageEmbeddings,
)

# Expose legacy embedding classes
from .legacy import (
    CharLMEmbeddings,
    DocumentLSTMEmbeddings,
    DocumentMeanEmbeddings,
    ELMoEmbeddings,
)

# Expose token embedding classes
from .token import (
    BytePairEmbeddings,
    CharacterEmbeddings,
    FastTextEmbeddings,
    FlairEmbeddings,
    HashEmbeddings,
    MuseCrosslingualEmbeddings,
    NILCEmbeddings,
    OneHotEmbeddings,
    PooledFlairEmbeddings,
    StackedEmbeddings,
    TokenEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)

__all__ = [
    "Embeddings",
    "ScalarMix",
    "DocumentCNNEmbeddings",
    "DocumentEmbeddings",
    "DocumentLMEmbeddings",
    "DocumentPoolEmbeddings",
    "DocumentRNNEmbeddings",
    "DocumentTFIDFEmbeddings",
    "SentenceTransformerDocumentEmbeddings",
    "TransformerDocumentEmbeddings",
    "ConvTransformNetworkImageEmbeddings",
    "IdentityImageEmbeddings",
    "ImageEmbeddings",
    "NetworkImageEmbeddings",
    "PrecomputedImageEmbeddings",
    "CharLMEmbeddings",
    "DocumentLSTMEmbeddings",
    "DocumentMeanEmbeddings",
    "BPEmbSerializable",
    "BytePairEmbeddings",
    "CharacterEmbeddings",
    "ELMoEmbeddings",
    "FastTextEmbeddings",
    "FlairEmbeddings",
    "HashEmbeddings",
    "MuseCrosslingualEmbeddings",
    "NILCEmbeddings",
    "OneHotEmbeddings",
    "PooledFlairEmbeddings",
    "StackedEmbeddings",
    "TokenEmbeddings",
    "TransformerWordEmbeddings",
    "WordEmbeddings",
    "TransformerEmbeddings",
    "TransformerOnnxWordEmbeddings",
    "TransformerOnnxDocumentEmbeddings",
    "TransformerJitWordEmbeddings",
    "TransformerJitDocumentEmbeddings",
]
