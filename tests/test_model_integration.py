import os
import shutil

from flair.data import Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings, TokenEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import SequenceTaggerTrainer, TextClassifierTrainer


def test_train_load_use_tagger():

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

    trainer.train('./results', learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_train_charlm_load_use_tagger():

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = CharLMEmbeddings('news-forward-fast')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train('./results', learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_train_charlm_changed_chache_load_use_tagger():

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    # make a temporary cache directory that we remove afterwards
    os.makedirs('./results/cache/', exist_ok=True)
    embeddings = CharLMEmbeddings('news-forward-fast', cache_directory='./results/cache/')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train('./results', learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    # remove the cache directory
    shutil.rmtree('./results/cache')

    loaded_model: SequenceTagger = SequenceTagger.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_train_charlm_nochache_load_use_tagger():

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = CharLMEmbeddings('news-forward-fast', use_cache=False)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train('./results', learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_load_use_serialized_tagger():

    loaded_model: SequenceTagger = SequenceTagger.load('ner')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])


def test_train_load_use_classifier():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

        loaded_model = TextClassifier.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_train_charlm_load_use_classifier():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: TokenEmbeddings = CharLMEmbeddings('news-forward-fast')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

        loaded_model = TextClassifier.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')


def test_train_charlm__nocache_load_use_classifier():
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: TokenEmbeddings = CharLMEmbeddings('news-forward-fast', use_cache=False)
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64,
                                                                         False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train('./results', max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

        loaded_model = TextClassifier.load_from_file('./results/final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree('./results')