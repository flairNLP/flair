"""This example is taken from https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md"""

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
dictionary: Dictionary = Dictionary.load("chars")

# get your corpus, process forward and at the character level
corpus = TextCorpus("/Users/aniket/data/penn_lm", dictionary, is_forward_lm, character_level=True)

if __name__== "__main__":
    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=128, nlayers=1)

    # train your language model
    trainer = LanguageModelTrainer(model=language_model, corpus=corpus, accelerator="cpu", devices=2)

    trainer.train(
        "resources/taggers/language_model",
        sequence_length=10,
        mini_batch_size=10,
        max_epochs=10,
        num_workers=2,
    )
