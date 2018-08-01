import shutil

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentMeanEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer


def test_training():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    model = TextClassifier([WordEmbeddings('en-glove')], 128, 1, False, 64, False, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    # clean up results directory
    shutil.rmtree('./results')
