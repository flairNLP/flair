import flair.datasets
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, FlairEmbeddings, WordEmbeddings, \
    SentenceTransformerDocumentEmbeddings, TransformerDocumentEmbeddings

import Evaluation
from em.EM_Clustering import EM_Clustering

if __name__ == '__main__':
    # embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
    embeddings = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

    labels = Evaluation.getStackOverFlowLabels()[1:20]
    documents = Evaluation.getStackOverFlowData()[1:20]
    listSentence = [Sentence(i) for i in documents]

    em = EM_Clustering(2, embeddings)
    em.cluster(listSentence, batchSize=1)

    predict = em.getDiskretResult()

    Evaluation.evaluate(labels, predict)
