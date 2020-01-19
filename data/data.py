import json

# documents = rosenberg.documents
#
# with open('/media/alpha/kanaan2/home/alpha/github/forks/flair/data/data.json', 'w+', encoding='utf8') as f:
#     f.write(json.dumps(documents))

with open('data.json', encoding='utf8') as f:
    data = json.loads(f.read())

data

from flair.datasets import JSONDataset

train_dataset: JSONDataset = JSONDataset('data.json', 'Beskrivning', ['Län', 'Härad', 'Tingslag', 'Församling', 'Plats'])

l = [ sentence.to_dict() for sentence in train_dataset.sentences]
l

train_dataset.sentences[:25]
sentence = train_dataset.sentences[0]
sentence.get_label_names()
sentence.to_original_text()


# TODO LazyRosenberg based on Flair...

# Access texts, documents, categories

# def _get_by_key(self, key):
#   return [elt[key] if key in elt else None for elt in self]

def _get_by_key(key):
    return [elt[key] if key in elt else None for elt in train_dataset.sentences]

import nltk

self = nltk.AbstractLazySequence
