from typing import List

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from flair.visual.training_curves import Plotter

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.UD_ENGLISH)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use contextual string embeddings
    #
    # CharLMEmbeddings('news-forward'),
    #
    # CharLMEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# initialize trainer
from flair.trainers.sequence_tagger_trainer import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

trainer.train('resources/taggers/example-ner', learning_rate=0.1, mini_batch_size=32, max_epochs=20)

plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')