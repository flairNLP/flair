from flair.file_utils import cached_path
from pathlib import Path
import gensim

for dim in [100, 400]:

    embeddings = f"normal_embeddings_{dim}d"

    cache_dir = Path("embeddings")

    # part of speech embeddings
    cached_path(f"glove_normal{dim}/vectors.txt", cache_dir=cache_dir)
    embeddings = cached_path(
        f"glove_normal{dim}/vectors.txt", cache_dir=cache_dir
    )

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        str(embeddings), binary=False
    )

    word_vectors.save(f"normal_embeddings_{dim}d")