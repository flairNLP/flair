from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CONLL_03, TREC_6
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models.multitask_model import EmbeddingSharedModel, SequenceTaggerTask, TextClassificationTask
from flair.models.multitask_model.utils import make_inputs
from flair.trainers import ModelTrainer

# tasks
conll03: Corpus = CONLL_03()
ner_dictionary = conll03.make_tag_dictionary('ner')
pos_dictionary = conll03.make_tag_dictionary('pos')

trec6: Corpus = TREC_6()
trec_dictionary = trec6.make_label_dictionary()

# embeddings
embedding_types = [
    WordEmbeddings('glove'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# models
ner_tagger: SequenceTaggerTask = SequenceTaggerTask(hidden_size=256,
                                                    embeddings=embeddings,
                                                    tag_dictionary=ner_dictionary,
                                                    tag_type='ner')

pos_tagger: SequenceTaggerTask = SequenceTaggerTask(hidden_size=256,
                                                    embeddings=embeddings,
                                                    tag_dictionary=pos_dictionary,
                                                    tag_type='pos')

trec_classifier: TextClassificationTask = TextClassificationTask()

multitask_corpora, multitask_models = make_inputs(
    {"corpus": conll03, "model": ner_tagger},
    {"corpus": conll03, "model": pos_tagger}
)

multitask_corpora: MultitaskCorpus = MultitaskCorpus(

)

multitask_model: EmbeddingSharedModel = EmbeddingSharedModel(ner_tagger, pos_tagger, trec_classifier)

trainer: ModelTrainer = ModelTrainer(multitask_model, multitask_corpora)
# 7. start training
trainer.train('folder/to/store',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)