from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

import Evaluation
from birch.Birch import Birch

if __name__ == '__main__':
    labels = Evaluation.getStackOverFlowLabels()[1:20]
    documents = Evaluation.getStackOverFlowData()[1:20]
    listSentence = [Sentence(i) for i in documents]

    embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

    birch = Birch(0.005, embedding, 3, 2)
    result = birch.cluster(listSentence)

    Evaluation.evaluate(labels, birch.predict)
