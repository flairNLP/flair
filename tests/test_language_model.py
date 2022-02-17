import pytest

from flair.data import Dictionary, Sentence
from flair.embeddings import FlairEmbeddings, TokenEmbeddings
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


@pytest.mark.integration
def test_train_language_model(results_base_path, resources_path):
    # get default dictionary
    dictionary: Dictionary = Dictionary.load("chars")

    # init forward LM with 128 hidden states and 1 layer
    language_model: LanguageModel = LanguageModel(dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)

    # get the example corpus and process at character level in forward direction
    corpus: TextCorpus = TextCorpus(
        resources_path / "corpora/lorem_ipsum",
        dictionary,
        language_model.is_forward_lm,
        character_level=True,
    )

    # train the language model
    trainer: LanguageModelTrainer = LanguageModelTrainer(language_model, corpus, test_mode=True)

    trainer.train(results_base_path, sequence_length=10, mini_batch_size=10, max_epochs=2)

    # use the character LM as embeddings to embed the example sentence 'I love Berlin'
    char_lm_embeddings: TokenEmbeddings = FlairEmbeddings(str(results_base_path / "best-lm.pt"))
    sentence = Sentence("I love Berlin")
    char_lm_embeddings.embed(sentence)

    text, likelihood = language_model.generate_text(number_of_characters=100)
    assert text is not None
    assert len(text) >= 100

    # clean up results directory
    del trainer, language_model, corpus, char_lm_embeddings


@pytest.mark.integration
def test_train_resume_language_model(resources_path, results_base_path, tasks_base_path):
    # get default dictionary
    dictionary: Dictionary = Dictionary.load("chars")

    # init forward LM with 128 hidden states and 1 layer
    language_model: LanguageModel = LanguageModel(dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)

    # get the example corpus and process at character level in forward direction
    corpus: TextCorpus = TextCorpus(
        resources_path / "corpora/lorem_ipsum",
        dictionary,
        language_model.is_forward_lm,
        character_level=True,
    )

    # train the language model
    trainer: LanguageModelTrainer = LanguageModelTrainer(language_model, corpus, test_mode=True)
    trainer.train(
        results_base_path,
        sequence_length=10,
        mini_batch_size=10,
        max_epochs=2,
        checkpoint=True,
    )
    del trainer, language_model

    trainer = LanguageModelTrainer.load_checkpoint(results_base_path / "checkpoint.pt", corpus)
    trainer.train(results_base_path, sequence_length=10, mini_batch_size=10, max_epochs=2)

    del trainer


def test_generate_text_with_small_temperatures():
    from flair.embeddings import FlairEmbeddings

    language_model = FlairEmbeddings("news-forward-fast").lm

    text, likelihood = language_model.generate_text(temperature=0.01, number_of_characters=100)
    assert text is not None
    assert len(text) >= 100
    del language_model


def test_compute_perplexity():
    from flair.embeddings import FlairEmbeddings

    language_model = FlairEmbeddings("news-forward-fast").lm

    grammatical = "The company made a profit"
    perplexity_gramamtical_sentence = language_model.calculate_perplexity(grammatical)

    ungrammatical = "Nook negh qapla!"
    perplexity_ungramamtical_sentence = language_model.calculate_perplexity(ungrammatical)

    print(f'"{grammatical}" - perplexity is {perplexity_gramamtical_sentence}')
    print(f'"{ungrammatical}" - perplexity is {perplexity_ungramamtical_sentence}')

    assert perplexity_gramamtical_sentence < perplexity_ungramamtical_sentence

    language_model = FlairEmbeddings("news-backward-fast").lm

    grammatical = "The company made a profit"
    perplexity_gramamtical_sentence = language_model.calculate_perplexity(grammatical)

    ungrammatical = "Nook negh qapla!"
    perplexity_ungramamtical_sentence = language_model.calculate_perplexity(ungrammatical)

    print(f'"{grammatical}" - perplexity is {perplexity_gramamtical_sentence}')
    print(f'"{ungrammatical}" - perplexity is {perplexity_ungramamtical_sentence}')

    assert perplexity_gramamtical_sentence < perplexity_ungramamtical_sentence
    del language_model
