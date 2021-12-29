from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, SentenceTransformerDocumentEmbeddings

import Evaluation
from kmeans.K_Means import KMeans

if __name__ == "__main__":
    # TODO: Fehler bei:
    # embedding = DocumentPoolEmbeddings([WordEmbeddings('glove')])
    # embedding = TransformerDocumentEmbeddings('bert-base-uncased')
    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    labels = Evaluation.get_stackoverflow_labels()[1:100]
    documents = Evaluation.get_stackoverflow_data()[1:100]
    sentences = [Sentence(i) for i in documents]

    kMeans = KMeans(5, embedding)
    result = kMeans.cluster(sentences)
    predict_labels = kMeans.get_label_list(sentences)

    Evaluation.evaluate(labels, predict_labels)
