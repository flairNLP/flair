from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

import evaluation
from birch.birch import Birch

if __name__ == "__main__":
    labels = evaluation.get_stackoverflow_labels()[1:20]
    documents = evaluation.get_stackoverflow_data()[1:20]
    sentences = [Sentence(i) for i in documents]

    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    birch = Birch(0.005, embedding, 3, 2, 5)
    result = birch.cluster(sentences)

    evaluation.evaluate(labels, birch.predict)
