import pytest
from torch.optim import Adam

import flair
from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentPoolEmbeddings, FlairEmbeddings, WordEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer

turian_embeddings = WordEmbeddings("turian")


@pytest.mark.integration()
def test_text_classifier_multi(results_base_path, tasks_base_path):
    flair.set_seed(123)

    flair_embeddings = FlairEmbeddings("news-forward-fast")

    corpus = ClassificationCorpus(
        tasks_base_path / "trivial" / "trivial_text_classification_single",
        label_type="city",
    )
    label_dict = corpus.make_label_dictionary(label_type="city")

    model: TextClassifier = TextClassifier(
        embeddings=DocumentPoolEmbeddings([flair_embeddings], fine_tune_mode="linear"),
        label_dictionary=label_dict,
        label_type="city",
    )

    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, mini_batch_size=2, max_epochs=1, shuffle=True)

    del model
    train_log_file = results_base_path / "training.log"
    assert train_log_file.exists()
    lines = train_log_file.read_text(encoding="utf-8").split("\n")
    expected_substrings = [
        "compute on device: ",
        "Corpus: ",
        "- learning_rate: ",
        "patience",
        "embedding storage:",
        "epoch 1 - iter",
        "EPOCH 1 done: loss",
        "Results:",
    ]
    for expected_substring in expected_substrings:
        assert any(expected_substring in line for line in lines), expected_substring


@pytest.mark.integration()
def test_train_load_use_tagger_large(results_base_path, tasks_base_path):
    corpus = flair.datasets.UD_ENGLISH().downsample(0.01)
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


@pytest.mark.integration()
def test_train_load_use_tagger_adam(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})
    tag_dictionary = corpus.make_label_dictionary("ner", add_unk=False)

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


def test_missing_validation_split(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(
        data_folder=tasks_base_path / "fewshot_conll",
        train_file="1shot.txt",
        sample_missing_splits=False,
        column_format={0: "text", 1: "ner"},
    )

    tag_dictionary = corpus.make_label_dictionary("ner", add_unk=True)

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
