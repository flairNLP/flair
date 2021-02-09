import flair
from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, StackedEmbeddings, CharacterEmbeddings
from flair.models.multitask_model import MultitaskModel, SequenceTaggerTask
from flair.trainers import ModelTrainer

import torch.nn

flair.device = "cuda:0"

"""
Example for training a multitask model in flair. Works similarly to the usual flow of flair.
1) Make Corpora and Tag / Label Dictionaries as usual. You can use one corpora for multiple tasks,
    by making two tag dictionaries.
2) You create your multitask model bottom-up, starting at low-level layers. Create a list of shared
    word embeddings and pass the either to stacked embeddings or any kind of document embeddings.
    They will use the shared word embedding layer but produce task specific outputs.
3) Create shared RNN or DocumentEmbedding layers that should be shared by different models. Pass them
    as arugments to your tasks.
4) Create your models and pass them shared embedding or shared rnn layers as to declared before.
5) Pass a dict in MultitaskCorpus class containing a mapping corpus - model. The class takes care of
    correctly assign a sentence to the correct forward_loss function in the MultitaskModel().
6) Pass MultitaskCorpus.models to MultitaskModel which dynamically creates your MultitaskModel.
7) Train as usual using our ModelTrainer by pass the MultitaskModel instance and MultitaskCorpus instance.
"""

# ----- CORPORA -----
conll03: Corpus = CONLL_03()

# ----- TAG SPACES -----
ner_dictionary = conll03.make_tag_dictionary('ner')
pos_dictionary = conll03.make_tag_dictionary('pos')

# ----- SHARED WORD EMBEDDING LAYER -----
shared_word_embeddings = [WordEmbeddings('glove'), CharacterEmbeddings()]
shared_word_embedding_layer: StackedEmbeddings = StackedEmbeddings(embeddings=shared_word_embeddings) # Stack if necessary

# ----- SHARED RNN LAYERS -----
shared_rnn_layer_labeling: torch.nn.Module = torch.nn.LSTM(input_size=shared_word_embedding_layer.embedding_length,
                                                           hidden_size=256,
                                                           num_layers=1,
                                                           bidirectional=True,
                                                           batch_first=True)

# ----- TASKS -----
ner_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=ner_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)

pos_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=pos_dictionary,
                                                    tag_type='pos',
                                                    use_rnn=True,
                                                    use_crf=True)

# ----- MULTITASK CORPUS -----
multi_corpus = MultitaskCorpus(
    {"corpus": conll03, "model": ner_tagger},
    {"corpus": conll03, "model": pos_tagger}
)

# ----- MULTITASK MODEL -----
multitask_model: MultitaskModel = MultitaskModel(multi_corpus.models)

# ----- TRAINING ON MODEL AND CORPUS -----
trainer: ModelTrainer = ModelTrainer(multitask_model, multi_corpus)
trainer.train('results/multitask-ner-pos',
              learning_rate=0.1,
              mini_batch_size=64,
              max_epochs=150,
              embeddings_storage_mode='gpu')