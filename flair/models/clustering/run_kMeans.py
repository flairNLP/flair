from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, SentenceTransformerDocumentEmbeddings

import evaluation
from kmeans.k_Means import KMeans

if __name__ == "__main__":
    # TODO: Fehler bei:
    # embedding = DocumentPoolEmbeddings([WordEmbeddings('glove')])
    # embedding = TransformerDocumentEmbeddings('bert-base-uncased')
    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    labels = evaluation.get_stackoverflow_labels()[1:500]
    documents = evaluation.get_stackoverflow_data()[1:500]
    sentences = [Sentence(i) for i in documents]

    kMeans = KMeans(20, embedding)
    result = kMeans.cluster(sentences)
    predict_labels = kMeans.get_label_list(sentences)

    evaluation.evaluate(labels, predict_labels)
