
from flair.datasets import WEBPAGES_NER
from flair.datasets import ICELANDIC_NER

corpus = WEBPAGES_NER()

print(corpus)
print(corpus.train[0])


# corpus = ICELANDIC_NER()
#
# print(corpus)
# print(corpus.train[0])