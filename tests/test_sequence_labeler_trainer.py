import shutil

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import SequenceTaggerTrainer


def test_training():

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = WordEmbeddings('glove')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train('./results', learning_rate=0.1, mini_batch_size=2, max_epochs=10)

    # clean up results directory
    shutil.rmtree('./results')
