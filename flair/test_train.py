from flair.datasets import CONLL_03

corpus = CONLL_03('D:\\flair\\resources')

tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')

# from flair.embeddings import TransformerWordEmbeddings
#
# embeddings = TransformerWordEmbeddings(
#     model='xlm-roberta-large',
#     layers="-1",
#     subtoken_pooling="first",
#     fine_tune=True,
#     use_context=True,
# )
from flair.embeddings import WordEmbeddings
embeddings = WordEmbeddings('glove')
from flair.trainers import ModelTrainer
from flair.models import BiaffineTager

tagger = BiaffineTager(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    use_biaffine=True,
    reproject_embeddings=False,
)
BiaffineTager.load()
from torch.optim.lr_scheduler import OneCycleLR

trainer = ModelTrainer(tagger, corpus)

trainer.train('D:\\flair\\resources\\test',
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=1,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              )