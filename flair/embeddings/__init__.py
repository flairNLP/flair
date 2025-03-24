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
    "BPEmbSerializable",
    "BytePairEmbeddings",
    "CharLMEmbeddings",
    "CharacterEmbeddings",
    "ConvTransformNetworkImageEmbeddings",
    "DocumentCNNEmbeddings",
    "DocumentEmbeddings",
    "DocumentLMEmbeddings",
    "DocumentLSTMEmbeddings",
    "DocumentMeanEmbeddings",
    "DocumentPoolEmbeddings",
    "DocumentRNNEmbeddings",
    "DocumentTFIDFEmbeddings",
    "ELMoEmbeddings",
    "Embeddings",
    "FastTextEmbeddings",
    "FlairEmbeddings",
    "HashEmbeddings",
    "IdentityImageEmbeddings",
    "ImageEmbeddings",
    "MuseCrosslingualEmbeddings",
    "NILCEmbeddings",
    "NetworkImageEmbeddings",
    "OneHotEmbeddings",
    "PooledFlairEmbeddings",
    "PrecomputedImageEmbeddings",
    "ScalarMix",
    "SentenceTransformerDocumentEmbeddings",
    "StackedEmbeddings",
    "TokenEmbeddings",
    "TransformerDocumentEmbeddings",
    "TransformerEmbeddings",
    "TransformerJitDocumentEmbeddings",
    "TransformerJitWordEmbeddings",
    "TransformerOnnxDocumentEmbeddings",
    "TransformerOnnxWordEmbeddings",
    "TransformerWordEmbeddings",
    "WordEmbeddings",
]
