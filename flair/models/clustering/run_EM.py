import flair.datasets
from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    WordEmbeddings,
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings,
)

import Evaluation
from em.EM_Clustering import EM_Clustering

if __name__ == "__main__":
    # embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
    embeddings = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    labels = Evaluation.get_stackoverflow_labels()[1:20]
    documents = Evaluation.get_stackoverflow_data()[1:20]
    sentences = [Sentence(i) for i in documents]

    em = EM_Clustering(2, embeddings)
    em.cluster(sentences, batch_size=1)

    predict = em.get_diskret_result()

    Evaluation.evaluate(labels, predict)
