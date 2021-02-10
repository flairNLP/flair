import flair
from flair.data import Corpus, MultitaskCorpus
from flair.datasets import BC2GM, NCBI_DISEASE, JNLPBA, BIOBERT_CHEMICAL_BC5CDR, BIOBERT_DISEASE_BC5CDR
from flair.embeddings import WordEmbeddings, StackedEmbeddings, CharacterEmbeddings
from flair.models.multitask_model import MultitaskModel, SequenceTaggerTask
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam

import torch.nn

flair.device = "cuda:2"

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
ncbi_disease: Corpus = NCBI_DISEASE()
bc2gm: Corpus = BC2GM()
bc5cdr_chemicals: Corpus = BIOBERT_CHEMICAL_BC5CDR()
bc5cdr_disease: Corpus = BIOBERT_DISEASE_BC5CDR()
jnlpba: Corpus = JNLPBA()

# ----- TAG SPACES -----
ncbi_dictionary = ncbi_disease.make_tag_dictionary('ner')
bc2gm_dictionary = bc2gm.make_tag_dictionary('ner')
bc5cdr_I_dictionary = bc5cdr_chemicals.make_tag_dictionary('ner')
bc5cdr_II_dictionary = bc5cdr_disease.make_tag_dictionary('ner')
jnlpba_dictionary = jnlpba.make_tag_dictionary('ner')

# ----- SHARED WORD EMBEDDING LAYER -----
shared_word_embeddings = [WordEmbeddings('pubmed'), CharacterEmbeddings(hidden_size_char=30)]
shared_word_embedding_layer: StackedEmbeddings = StackedEmbeddings(embeddings=shared_word_embeddings) # Stack if necessary

# ----- SHARED RNN LAYERS -----
shared_rnn_layer_labeling: torch.nn.Module = torch.nn.LSTM(input_size=shared_word_embedding_layer.embedding_length,
                                                           hidden_size=256,
                                                           num_layers=1,
                                                           bidirectional=True,
                                                           batch_first=True)

# ----- TASKS -----
ncbi_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=ncbi_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)

bc2gm_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=bc2gm_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)

bc5cdr_I_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=bc5cdr_I_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)

bc5cdr_II_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=bc5cdr_II_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)

jnlpba_tagger: SequenceTaggerTask = SequenceTaggerTask(embeddings=shared_word_embedding_layer,
                                                    tag_dictionary=jnlpba_dictionary,
                                                    tag_type='ner',
                                                    rnn=shared_rnn_layer_labeling,
                                                    use_crf=True)



# ----- MULTITASK CORPUS -----
multi_corpus = MultitaskCorpus(
    {"corpus": ncbi_disease, "model": ncbi_tagger},
    {"corpus": bc2gm, "model": bc2gm_tagger},
    {"corpus": bc5cdr_chemicals, "model": bc5cdr_I_tagger},
    {"corpus": bc5cdr_disease, "model": bc5cdr_II_tagger},
    {"corpus": jnlpba, "model": jnlpba_tagger}
)

# ----- MULTITASK MODEL -----
multitask_model: MultitaskModel = MultitaskModel(multi_corpus.models)

# ----- TRAINING ON MODEL AND CORPUS -----
trainer: ModelTrainer = ModelTrainer(multitask_model, multi_corpus)
trainer.train('results/multitask-bio',
              learning_rate=0.01,
              mini_batch_size=64,
              max_epochs=150,
              embeddings_storage_mode='gpu')