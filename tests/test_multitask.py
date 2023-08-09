import pytest

import flair
from flair.data import Sentence
from flair.datasets import SENTEVAL_CR, SENTEVAL_SST_GRANULAR
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import MultitaskModel, TextClassifier
from flair.nn.multitask import make_multitask_model_and_corpus
from flair.trainers import ModelTrainer


@pytest.mark.integration()
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    # --- Embeddings that are shared by both models --- #
    shared_embedding = TransformerDocumentEmbeddings("distilbert-base-uncased", fine_tune=True)

    # --- Task 1: Sentiment Analysis (5-class) --- #
    flair.set_seed(123)

    # Define corpus and model
    corpus_1 = SENTEVAL_SST_GRANULAR().downsample(0.01)

    model_1 = TextClassifier(
        shared_embedding, label_dictionary=corpus_1.make_label_dictionary("class", add_unk=False), label_type="class"
    )

    # -- Task 2: Binary Sentiment Analysis on Customer Reviews -- #
    flair.set_seed(123)

    # Define corpus and model
    corpus_2 = SENTEVAL_CR().downsample(0.01)

    model_2 = TextClassifier(
        shared_embedding,
        label_dictionary=corpus_2.make_label_dictionary("sentiment", add_unk=False),
        label_type="sentiment",
        inverse_model=True,
    )

    # -- Define mapping (which tagger should train on which model) -- #
    multitask_model, multicorpus = make_multitask_model_and_corpus(
        [
            (model_1, corpus_1),
            (model_2, corpus_2),
        ]
    )

    # -- Create model trainer and train -- #
    trainer = ModelTrainer(multitask_model, multicorpus)

    trainer.fine_tune(results_base_path, max_epochs=1)

    del trainer, multitask_model, corpus_1, corpus_2
    loaded_model = MultitaskModel.load(results_base_path / "final-model.pt")

    sentence = Sentence("I love Berlin")
    sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])

    for label in sentence.labels:
        assert label.value is not None
        assert 0.0 <= label.score <= 1.0
        assert isinstance(label.score, float)
    del loaded_model
