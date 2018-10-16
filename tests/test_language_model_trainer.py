import shutil

from flair.data import Dictionary, Sentence
from flair.embeddings import CharLMEmbeddings
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


def test_training():
    # get default dictionary
    dictionary: Dictionary = Dictionary.load('chars')

    # init forward LM with 128 hidden states and 1 layer
    language_model: LanguageModel = LanguageModel(dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)

    # get the example corpus and process at character level in forward direction
    corpus: TextCorpus = TextCorpus('resources/corpora/lorem_ipsum',
                                    dictionary,
                                    language_model.is_forward_lm,
                                    character_level=True)

    # train the language model
    trainer: LanguageModelTrainer = LanguageModelTrainer(language_model, corpus)
    trainer.train('./results', sequence_length=10, mini_batch_size=10, max_epochs=5)

    # use the character LM as embeddings to embed the example sentence 'I love Berlin'
    char_lm_embeddings = CharLMEmbeddings('./results/best-lm.pt')
    sentence = Sentence('I love Berlin')
    char_lm_embeddings.embed(sentence)
    print(sentence[1].embedding.size())

    # clean up results directory
    shutil.rmtree('./results', ignore_errors=True)


