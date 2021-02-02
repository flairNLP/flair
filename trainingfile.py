from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CONLL_03, TREC_6
from flair.embeddings import WordEmbeddings, StackedEmbeddings, DocumentRNNEmbeddings, CharacterEmbeddings
from flair.models.multitask_model import MultitaskModel, SequenceTaggerTask, TextClassificationTask
from flair.trainers import ModelTrainer

# tasks
conll03: Corpus = CONLL_03()
ner_dictionary = conll03.make_tag_dictionary('ner')
pos_dictionary = conll03.make_tag_dictionary('pos')

#trec6: Corpus = TREC_6()
#trec_dictionary = trec6.make_label_dictionary()

# embeddings
embeddings = [
    WordEmbeddings('glove'),
    CharacterEmbeddings()
]

word_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

#document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(embeddings)

ner_tagger: SequenceTaggerTask = SequenceTaggerTask(hidden_size=256,
                                                    embeddings=word_embeddings,
                                                    tag_dictionary=ner_dictionary,
                                                    tag_type='ner',
                                                    use_crf=True)

pos_tagger: SequenceTaggerTask = SequenceTaggerTask(hidden_size=256,
                                                    embeddings=word_embeddings,
                                                    tag_dictionary=pos_dictionary,
                                                    tag_type='pos',
                                                    use_crf=True)
"""
trec_classifier: TextClassificationTask = TextClassificationTask(document_embeddings,
                                                                 label_dictionary=trec_dictionary)
"""

multi_corpus = MultitaskCorpus(
    {"corpus": conll03, "model": ner_tagger},
    {"corpus": conll03, "model": pos_tagger}
)

multitask_model: MultitaskModel = MultitaskModel(multi_corpus.models)

trainer: ModelTrainer = ModelTrainer(multitask_model, multi_corpus)
# 7. start training
trainer.train('folder/to/store',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)