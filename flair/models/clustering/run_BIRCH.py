import evaluation
from birch.birch import Birch
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

if __name__ == "__main__":
    labels = evaluation.get_stackoverflow_labels()[1:40]
    documents = evaluation.get_stackoverflow_data()[1:40]
    sentences = [Sentence(i) for i in documents]

    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    birch = Birch(0.005, embedding, 4, 3, 5)
    result = birch.cluster(sentences)

    evaluation.evaluate(labels, birch.predict)
