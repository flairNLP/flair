import shutil

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import CharLMEmbeddings, WordEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer


def test_training():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)

    charlm_embedding_forward = CharLMEmbeddings('news-forward')
    charlm_embedding_backward = CharLMEmbeddings('news-backward')
    glove_embedding = WordEmbeddings('en-glove')

    label_dict = corpus.make_label_dictionary()

    model = TextClassifier([charlm_embedding_forward, charlm_embedding_backward, glove_embedding], 128, 1, False, False, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)

    trainer.train('./results', max_epochs=5)

    shutil.rmtree('./results')
