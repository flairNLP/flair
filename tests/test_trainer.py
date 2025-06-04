import pytest
from torch.optim import Adam
import shutil  # Import shutil for cleanup
from pathlib import Path  # Import Path for type hinting

import flair
from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentPoolEmbeddings, FlairEmbeddings, WordEmbeddings
from flair.models import SequenceTagger, TextClassifier, TokenClassifier
from flair.tokenization import StaccatoTokenizer, SegtokTokenizer, Tokenizer
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


@pytest.mark.integration()
def test_store_tokenizer_in_model(results_base_path, tasks_base_path):
    """Tests if the tokenizer used in the corpus is stored in the model."""
    flair.set_seed(123)

    # 1. Get a corpus and instantiate with custom tokenizer
    tokenizer = StaccatoTokenizer()
    corpus = flair.datasets.ColumnCorpus(
        data_folder=tasks_base_path / "fewshot_conll",
        train_file="1shot.txt",
        sample_missing_splits=False,
        column_format={0: "text", 1: "ner"},
        use_tokenizer=tokenizer,
    )

    # 2. Make label dictionary
    label_dict = corpus.make_label_dictionary(label_type="ner")

    # 3. Initialize model
    model = TokenClassifier(
        embeddings=turian_embeddings,
        label_dictionary=label_dict,
        label_type="ner",
    )

    # 4. Initialize trainer
    trainer = ModelTrainer(model, corpus)

    # 5. Train briefly
    trainer.train(
        results_base_path / "test_store_tokenizer",
        learning_rate=0.01,
        mini_batch_size=1,
        max_epochs=1,
        shuffle=False,
    )

    # 6. Assert that the tokenizer is stored in the model
    assert hasattr(model, "_tokenizer")
    assert model._tokenizer is not None
    assert isinstance(model._tokenizer, StaccatoTokenizer)

    # Clean up
    del trainer, model, label_dict, corpus, tokenizer


# --- New Tests for Save/Load ---


def _train_and_load_model_with_tokenizer(
    results_base_path: Path,
    tasks_base_path: Path,
    tokenizer: Tokenizer,
    model_class=TokenClassifier,
    embeddings=turian_embeddings,
) -> flair.nn.Model:
    """Helper function to train a model with a specific tokenizer, reload it, and clean up."""
    flair.set_seed(123)
    # Define the path for saving the model specific to this test run
    model_save_path = results_base_path / f"test_{tokenizer.__class__.__name__}_save_load_{tokenizer.name}"
    loaded_model = None  # Initialize loaded_model

    try:
        # 1. Get a corpus and instantiate with custom tokenizer
        corpus = flair.datasets.ColumnCorpus(
            data_folder=tasks_base_path / "fewshot_conll",  # Use a small dataset
            train_file="1shot.txt",
            sample_missing_splits=False,
            column_format={0: "text", 1: "ner"},
            use_tokenizer=tokenizer,  # Pass the specific tokenizer
        ).downsample(
            1.0
        )  # Use the full 1shot file

        # 2. Make label dictionary
        label_dict = corpus.make_label_dictionary(label_type="ner")

        # 3. Initialize model
        if model_class == TokenClassifier:
            model = TokenClassifier(
                embeddings=embeddings,
                label_dictionary=label_dict,
                label_type="ner",
            )
        # Add other model types here if needed for testing
        else:
            raise NotImplementedError(f"Model class {model_class} not supported in helper.")

        # 4. Initialize trainer
        trainer = ModelTrainer(model, corpus)

        # 5. Train briefly
        trainer.train(
            model_save_path,
            learning_rate=0.01,
            mini_batch_size=1,
            max_epochs=1,
            shuffle=False,
            save_final_model=True,  # Ensure model is saved
        )

        # 6. Load the saved model
        final_model_file = model_save_path / "final-model.pt"
        if final_model_file.exists():
            loaded_model = model_class.load(final_model_file)
        else:
            pytest.fail(f"Model training did not produce 'final-model.pt' at {model_save_path}")

        # Clean up intermediate objects (optional within try)
        del trainer, model, label_dict, corpus

    finally:
        # 7. Cleanup: Remove the directory created for this test run
        if model_save_path.exists():
            shutil.rmtree(model_save_path)

    return loaded_model  # Return the loaded model


@pytest.mark.integration()
def test_staccato_tokenizer_save_load(results_base_path, tasks_base_path):
    """Tests if StaccatoTokenizer is correctly saved and loaded with the model."""
    tokenizer = StaccatoTokenizer()
    loaded_model = _train_and_load_model_with_tokenizer(results_base_path, tasks_base_path, tokenizer)

    # Assertions
    assert hasattr(loaded_model, "_tokenizer")
    assert loaded_model._tokenizer is not None
    assert isinstance(loaded_model._tokenizer, StaccatoTokenizer)

    # Optional: Check if tokenization works as expected
    sentence = Sentence("Test sentence.")
    tokens = loaded_model._tokenizer.tokenize(sentence.text)  # Use tokenizer directly
    assert tokens == ["Test", "sentence", "."]  # Check Staccato tokenization result

    del loaded_model, tokenizer


@pytest.mark.integration()
def test_segtok_tokenizer_save_load(results_base_path, tasks_base_path):
    """Tests if SegtokTokenizer is correctly saved and loaded with the model."""
    # Test default Segtok
    tokenizer_default = SegtokTokenizer()
    loaded_model_default = _train_and_load_model_with_tokenizer(results_base_path, tasks_base_path, tokenizer_default)

    assert hasattr(loaded_model_default, "_tokenizer")
    assert loaded_model_default._tokenizer is not None
    assert isinstance(loaded_model_default._tokenizer, SegtokTokenizer)
    assert loaded_model_default._tokenizer.additional_split_characters is None  # Check default param

    # Test Segtok with custom split chars
    split_chars = ["§", "."]
    tokenizer_custom = SegtokTokenizer(additional_split_characters=split_chars)
    loaded_model_custom = _train_and_load_model_with_tokenizer(results_base_path, tasks_base_path, tokenizer_custom)

    assert hasattr(loaded_model_custom, "_tokenizer")
    assert loaded_model_custom._tokenizer is not None
    assert isinstance(loaded_model_custom._tokenizer, SegtokTokenizer)
    assert loaded_model_custom._tokenizer.additional_split_characters == split_chars  # Check custom param restored

    # Optional: Check functionality difference (though not strictly testing save/load)
    text = "Test.Symbol"
    tokens_default = loaded_model_default._tokenizer.tokenize(text)
    tokens_custom = loaded_model_custom._tokenizer.tokenize(text)
    assert "Test.Symbol" in tokens_default  # Default doesn't split '.'
    assert "." in tokens_custom  # Custom should split '.'

    del loaded_model_default, tokenizer_default
    del loaded_model_custom, tokenizer_custom
