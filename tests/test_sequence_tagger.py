import pytest
from torch.optim import SGD
from torch.optim.adam import Adam

import flair.datasets
from flair.data import MultiCorpus, Sentence
from flair.embeddings import FlairEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

turian_embeddings = WordEmbeddings("turian")
flair_embeddings = FlairEmbeddings("news-forward-fast")


@pytest.mark.integration
def test_load_use_tagger():
    loaded_model: SequenceTagger = SequenceTagger.load("ner")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    del loaded_model

    sentence.clear_embeddings()
    sentence_empty.clear_embeddings()

    loaded_model: SequenceTagger = SequenceTagger.load("pos")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    del loaded_model


@pytest.mark.integration
def test_load_use_tagger_keep_embedding():
    loaded_model: SequenceTagger = SequenceTagger.load("ner")

    sentence = Sentence("I love Berlin")
    loaded_model.predict(sentence)
    for token in sentence:
        assert len(token.embedding.cpu().numpy()) == 0

    loaded_model.predict(sentence, embedding_storage_mode="cpu")
    for token in sentence:
        assert len(token.embedding.cpu().numpy()) > 0

    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, tag_dictionary, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    # clean up results directory
    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger_empty_tags(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 2: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, tag_dictionary, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger_disjunct_tags(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(
        data_folder=tasks_base_path / "fashion_disjunct",
        column_format={0: "text", 3: "ner"},
    )
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
        allow_unk_predictions=True,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )


@pytest.mark.integration
def test_train_load_use_tagger_large(results_base_path, tasks_base_path):
    corpus = flair.datasets.UD_ENGLISH().downsample(0.05)
    tag_dictionary = corpus.make_label_dictionary("pos")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="pos",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, tag_dictionary, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger_flair_embeddings(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=flair_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, tag_dictionary, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger_adam(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
        optimizer=Adam,
    )

    del trainer, tagger, tag_dictionary, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_load_use_tagger_multicorpus(results_base_path, tasks_base_path):
    corpus_1 = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    corpus_2 = flair.datasets.NER_GERMAN_GERMEVAL(base_path=tasks_base_path).downsample(0.1)

    corpus = MultiCorpus([corpus_1, corpus_2])
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
        allow_unk_predictions=True,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, corpus
    loaded_model: SequenceTagger = SequenceTagger.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    del loaded_model


@pytest.mark.integration
def test_train_resume_tagger(results_base_path, tasks_base_path):

    corpus_1 = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    corpus_2 = flair.datasets.NER_GERMAN_GERMEVAL(base_path=tasks_base_path).downsample(0.1)

    corpus = MultiCorpus([corpus_1, corpus_2])
    tag_dictionary = corpus.make_label_dictionary("ner")

    model: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # train model for 2 epochs
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False, checkpoint=True)

    del model

    # load the checkpoint model and train until epoch 4
    checkpoint_model = SequenceTagger.load(results_base_path / "checkpoint.pt")
    trainer.resume(model=checkpoint_model, max_epochs=4)

    # clean up results directory
    del trainer


@pytest.mark.integration
def test_find_learning_rate(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.find_learning_rate(results_base_path, optimizer=SGD, iterations=5)

    del trainer, tagger, tag_dictionary, corpus
