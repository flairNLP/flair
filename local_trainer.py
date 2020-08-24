import flair.datasets
from flair.embeddings import POSEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = flair.datasets.WIKINER_ENGLISH().downsample(0.0001)

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types = [
    POSEmbeddings(),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/ner-tagger-with-pos-embeddings',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)