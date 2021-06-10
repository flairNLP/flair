from flair.models import SequenceTagger
from flair.data import Sentence

sentence = Sentence('I love Berlin .')
tagger = SequenceTagger.load('flair/ner-english')
print(tagger.push_to_hub('flair-ner-english3', commit_message="This is a new model"))
