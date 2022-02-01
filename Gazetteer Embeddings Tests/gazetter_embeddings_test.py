from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, GazetteerEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# corpus = CONLL_03()
# label_type = 'ner'
# label_dict = corpus.make_label_dictionary(label_type=label_type)

sentence = Sentence('The grass is green .')
#
# transformer_embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
#                                                    layers="-1",
#                                                    subtoken_pooling="first",
#                                                    fine_tune=True,
#                                                    use_context=True,
#                                                    )
#
# transformer_embeddings.embed(sentence)
#
# for token in sentence:
#     print(token)
#     print(token.embedding)

##########The Goal############

# # somehow load a gazetteer
# loaded_gazetteer = load(...)
#
# # init embedding with gazetteer
gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(gazetteers=["test"])

print(gazetteer_embedding.matching_methods)
#
gazetteer_embedding.embed(sentence)

for token in sentence:
    print(token)
    print(token.embedding)

# embeddings_list = [gazetteer_embedding, transformer_embeddings]
#
# stacked_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings_list)
#
# sentence = Sentence('The grass is not green .')
#
# stacked_embeddings.embed(sentence)
#
# for token in sentence:
#     print(token)
#     print(token.embedding)
