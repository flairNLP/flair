import pytest

import flair
from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentPoolEmbeddings, FlairEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


@pytest.mark.integration
def test_text_classifier_multi(results_base_path, tasks_base_path):
    flair.set_seed(123)

    flair_embeddings = FlairEmbeddings("news-forward-fast")

    corpus = ClassificationCorpus(
        tasks_base_path / "trivial" / "trivial_text_classification_single",
        label_type="city",
    )
    label_dict = corpus.make_label_dictionary(label_type="city")

    model: TextClassifier = TextClassifier(
        document_embeddings=DocumentPoolEmbeddings([flair_embeddings], fine_tune_mode="linear"),
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
        "Device: ",
        "Corpus: ",
        "Parameters:",
        "- learning_rate: ",
        "- patience: ",
        "Embeddings storage mode:",
        "epoch 1 - iter",
        "EPOCH 1 done: loss",
        "Results:",
    ]
    for expected_substring in expected_substrings:
        assert any(expected_substring in line for line in lines), expected_substring
