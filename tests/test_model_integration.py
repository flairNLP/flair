import os
import shutil
import pytest

from flair.data import Dictionary, Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings, TokenEmbeddings
from flair.models import SequenceTagger, TextClassifier, LanguageModel
from flair.trainers import SequenceTaggerTrainer, TextClassifierTrainer
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


@pytest.mark.slow
def test_train_load_use_tagger(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = WordEmbeddings('glove')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train(str(results_base_path), learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_charlm_load_use_tagger(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = CharLMEmbeddings('news-forward-fast')

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train(str(results_base_path), learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_charlm_changed_chache_load_use_tagger(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    # make a temporary cache directory that we remove afterwards
    cache_dir = results_base_path / 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    embeddings = CharLMEmbeddings('news-forward-fast', cache_directory=cache_dir)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train(str(results_base_path), learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    # remove the cache directory
    shutil.rmtree(cache_dir)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_charlm_nochache_load_use_tagger(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary('ner')

    embeddings = CharLMEmbeddings('news-forward-fast', use_cache=False)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=False)

    # initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    trainer.train(str(results_base_path), learning_rate=0.1, mini_batch_size=2, max_epochs=3)

    loaded_model: SequenceTagger = SequenceTagger.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


def test_load_use_serialized_tagger():

    loaded_model: SequenceTagger = SequenceTagger.load('ner')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    sentence.clear_embeddings()
    sentence_empty.clear_embeddings()

    loaded_model: SequenceTagger = SequenceTagger.load('pos')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])


@pytest.mark.slow
def test_train_load_use_classifier(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB, base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings('en-glove')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train(str(results_base_path), max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

    loaded_model = TextClassifier.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_charlm_load_use_classifier(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB, base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: TokenEmbeddings = CharLMEmbeddings('news-forward-fast')
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64, False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train(str(results_base_path), max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

    loaded_model = TextClassifier.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_charlm_nocache_load_use_classifier(results_base_path, tasks_base_path):

    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB, base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()

    glove_embedding: TokenEmbeddings = CharLMEmbeddings('news-forward-fast', use_cache=False)
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove_embedding], 128, 1, False, 64,
                                                                         False,
                                                                         False)

    model = TextClassifier(document_embeddings, label_dict, False)

    trainer = TextClassifierTrainer(model, corpus, label_dict, False)
    trainer.train(str(results_base_path), max_epochs=2)

    sentence = Sentence("Berlin is a really nice city.")

    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)

    loaded_model = TextClassifier.load_from_file(results_base_path / 'final-model.pt')

    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)


@pytest.mark.slow
def test_train_language_model(results_base_path, resources_path):
    # get default dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # init forward LM with 128 hidden states and 1 layer
    language_model: LanguageModel = LanguageModel(dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)

    # get the example corpus and process at character level in forward direction
    corpus: TextCorpus = TextCorpus(str(resources_path / 'corpora/lorem_ipsum'),
                                    dictionary,
                                    language_model.is_forward_lm,
                                    character_level=True)

    # train the language model
    trainer: LanguageModelTrainer = LanguageModelTrainer(language_model, corpus)
    trainer.train(str(results_base_path), sequence_length=10, mini_batch_size=10, max_epochs=5)

    # use the character LM as embeddings to embed the example sentence 'I love Berlin'
    char_lm_embeddings = CharLMEmbeddings(str(results_base_path / 'best-lm.pt'))
    sentence = Sentence('I love Berlin')
    char_lm_embeddings.embed(sentence)

    # clean up results directory
    shutil.rmtree(results_base_path, ignore_errors=True)
