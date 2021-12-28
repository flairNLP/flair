from flair.data import Sentence
from flair.embeddings import WordEmbeddings, \
    DocumentPoolEmbeddings, SentenceTransformerDocumentEmbeddings

import Evaluation
from kmeans.K_Means import KMeans

if __name__ == '__main__':
    # TODO: Fehler bei:
    # embedding = DocumentPoolEmbeddings([WordEmbeddings('glove')])
    # embedding = TransformerDocumentEmbeddings('bert-base-uncased')
    embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

    labels = Evaluation.getStackOverFlowLabels()[1:100]
    documents = Evaluation.getStackOverFlowData()[1:100]
    listSentence = [Sentence(i) for i in documents]

    kMeans = KMeans(5, embedding)
    result = kMeans.cluster(listSentence)
    predict_labels = kMeans.getLabelList(listSentence)

    Evaluation.evaluate(labels, predict_labels)
