import flair.datasets
from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    WordEmbeddings,
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings,
)

import evaluation
from em.EM_Clustering import EM_Clustering

if __name__ == "__main__":
    # embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
    embeddings = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    labels = evaluation.get_20_news_label()[1:300]
    documents = evaluation.get_20news_data()[1:300]
    sentences = [Sentence(i) for i in documents]

    em = EM_Clustering(15, embeddings)
    em.cluster(sentences, batch_size=1)

    predict = em.get_discrete_result()
    evaluation.evaluate(labels, predict)
