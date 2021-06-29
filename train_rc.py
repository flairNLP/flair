import torch.optim

import flair.datasets
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings

# 1. get the corpus
from flair.models.relation_classifier_model import RelationClassifierLinear

corpus: Corpus = flair.datasets.SEMEVAL_2010_TASK_8(in_memory=False).downsample(0.1)
print(corpus.train[1])

label_dictionary = corpus.make_label_dictionary("relation")

# initialize embeddings
# embeddings = TransformerWordEmbeddings(layers="-1", fine_tune=True)

# initialize sequence tagger
# model: RelationClassifierLinear = RelationClassifierLinear(
#     token_embeddings=embeddings,
#     label_dictionary=label_dictionary,
#     label_type="relation",
#     span_label_type="ner",
# )
#
# # initialize trainer
# from flair.trainers import ModelTrainer
#
# # initialize trainer
# trainer: ModelTrainer = ModelTrainer(model, corpus, optimizer=torch.optim.Adam)
#
# trainer.train(
#     "resources/classifiers/example-rc-linear",
#     learning_rate=3e-5,
#     mini_batch_size=4,
#     mini_batch_chunk_size=1,
#     max_epochs=10,
#     shuffle=True,
# )

model = RelationClassifierLinear.load("resources/classifiers/example-rc-linear/best-model.pt")
result, score = model.evaluate(corpus.test)

print(result.detailed_results)