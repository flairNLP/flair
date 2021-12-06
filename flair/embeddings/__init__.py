# Expose base classses
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
    BertEmbeddings,
    CamembertEmbeddings,
    CharLMEmbeddings,
    DocumentLSTMEmbeddings,
    DocumentMeanEmbeddings,
    ELMoTransformerEmbeddings,
    OpenAIGPT2Embeddings,
    OpenAIGPTEmbeddings,
    RoBERTaEmbeddings,
    TransformerXLEmbeddings,
    XLMEmbeddings,
    XLMRobertaEmbeddings,
    XLNetEmbeddings,
)

# Expose token embedding classes
from .token import (
    BPEmbSerializable,
    BytePairEmbeddings,
    CharacterEmbeddings,
    ELMoEmbeddings,
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
