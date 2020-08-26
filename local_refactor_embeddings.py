from flair.file_utils import cached_path
from pathlib import Path
import gensim

for dim in [100, 400]:

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        f"glove_normal{dim}/vectors.txt", binary=False
    )

    word_vectors.save(f"normal_embeddings_{dim}d")