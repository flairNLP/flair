from flair.file_utils import cached_path
from pathlib import Path
import gensim

hu_path: str = "https://flair.informatik.hu-berlin.de/resources"

for dim in [25, 100]:

    embeddings = f"pos_embeddings_{dim}d"

    hu_path: str = "https://flair.informatik.hu-berlin.de/resources"

    cache_dir = Path("embeddings")

    # part of speech embeddings
    cached_path(f"{hu_path}/embeddings/{embeddings}.txt", cache_dir=cache_dir)
    embeddings = cached_path(
        f"{hu_path}/embeddings/{embeddings}.txt", cache_dir=cache_dir
    )

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        str(embeddings), binary=False
    )

    word_vectors.save(f"pos_embeddings_{dim}d")