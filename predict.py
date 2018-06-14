from flair.data import Sentence
from flair.tagging_model import SequenceTaggerLSTM
import torch

model_path: str = 'resources/taggers/example-pos'

tagger: SequenceTaggerLSTM = torch.load(model_path + '/model.pt', map_location={'cuda:0': 'cpu'})
tagger.eval()
if torch.cuda.is_available():
    tagger = tagger.cuda()

def predict(text: str):
    sentence: Sentence = Sentence(text)
    tagger.predict(sentence, tag_type='upos')
    print(sentence.to_ner_string())

predict('George Washington was born in Vermont .')