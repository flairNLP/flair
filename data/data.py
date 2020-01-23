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

data_dict = [
    {
        'Plats': 'Abbotnäs',
        'Församling': 'Floda',
        'Härad': 'Oppunda',
        'Län': 'Södermanlands län',
        'Beskrivning': 'Abbotnäs. Herrgård i Floda sn, Oppunda hd, Södermanlands län, n. om Valla jernvägsstation vid den lilla sjön Floden. (Skall ha tillhört Julita kloster.) Godset utgöres af Abbotnäs säteri 1 1/2 mtl, Båresta 1 7/8, Brunvalla 1, Baggetorp 5/8, Buddetorp 1/2, Heden eller Staffanstorp 1/8, Hvinstugan 1/4, Löta 1, Munkeboda 1/8, Sjölunda 1/2, Stafhälla 1/2, tills. 8 mtl., tax. 145,000 kr.; såg 2,000 kr.'
    }
]

data_json = json.dumps(data_dict)

train_dataset: JSONDataset = JSONDataset(data_json, 'Beskrivning', ['Län', 'Härad', 'Tingslag', 'Församling', 'Plats'])

l = [ sentence.to_dict() for sentence in train_dataset.sentences]
l

import io

data_stream = io.StringIO(data_json)

train_dataset: JSONDataset = JSONDataset(data_stream, 'Beskrivning', ['Län', 'Härad', 'Tingslag', 'Församling', 'Plats'])

l = [ sentence.to_dict() for sentence in train_dataset.sentences]
l



# TODO LazyRosenberg(Dataset) based on Flair...
# TODO sents <- sentences, words <- tokens, etc

# Access texts, documents, categories

# def _get_by_key(self, key):
#   return [elt[key] if key in elt else None for elt in self]

def _get_by_key(key):
    return [elt[key] if key in elt else None for elt in train_dataset.sentences]


def hundreds():
    h1 = self._get_by_key('Härad')
    h2 = self._get_by_key('Tingslag')
    return self._get_values(self._hundreds, [_1 if _2 is None else _2 for _1, _2 in zip(h1, h2)])



import nltk

self = nltk.AbstractLazySequence



#---

import pandas as pd

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

df = pd.DataFrame(zip((twenty_train.target_names[_] for _ in twenty_train.target), twenty_train.data), columns=('category', 'text'))
# df.head()

records = df.to_dict(orient='records')

data_json = json.dumps(records)
train_dataset: JSONDataset = JSONDataset(data_json, 'text', ['category'])

train_dataset.categories
# ['category']

train_dataset.total_sentence_count
# 2257

sents = train_dataset.sentences

sents[0]
# Sentence: "From : sd345 @ city.ac.uk ( Michael Collier ) Subject : Converting images to HP LaserJet III ? Nntp-Posting-Host : hampton Organization : The City ... ." - 125 Tokens

sents[0].labels
# [comp.graphics (1.0)]

import io

data_stream = io.StringIO(data_json)
train_dataset: JSONDataset = JSONDataset(data_stream, 'text', ['category'])

train_dataset.total_sentence_count
# 2257

with open('20news.json', 'w+') as f:
    f.write(data_json)

# with open('20news.json') as f:
#     data_json = f.read()

train_dataset: JSONDataset = JSONDataset('20news.json', 'text', ['category'])

train_dataset.total_sentence_count
# 2257
