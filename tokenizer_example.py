from flair.data import LabeledString

# LabeledString is a DataPoint - init and set the label
from flair.models.tokenizer_model import FlairTokenizer

sentence = LabeledString('Any major dischord and we all suffer.')
sentence.set_label('tokenization', 'BIEXBIIIEXBIIIIIIEXBIEXBEXBIEXBIIIIES')

# Print the DataPoint
print(sentence)

# Print the string
print(sentence.string)

# print the label
print(sentence.get_labels('tokenization'))

# init the tokenizer
tokenizer: FlairTokenizer = FlairTokenizer()

# do a forward pass and compute the loss for the data point
tokenizer.forward_loss(sentence)