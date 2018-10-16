import shutil

from flair.data import Sentence

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentMeanEmbeddings, DocumentLSTMEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer


def test_text_classifier_single_label():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False, False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert(l.value is not None)
            assert(0.0 <= l.score <= 1.0)
            assert(type(l.score) is float)

    # clean up results directory
    shutil.rmtree('./results')


def test_text_classifier_mulit_label():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentMeanEmbeddings = DocumentMeanEmbeddings([glove_embedding])

    model = TextClassifier(document_embeddings, label_dict, True)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert(l.value is not None)
            assert(0.0 <= l.score <= 1.0)
            assert(type(l.score) is float)

    # clean up results directory
    shutil.rmtree('./results')