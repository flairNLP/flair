from flair.data import LabeledString

# LabeledString is a DataPoint - init and set the label
sentence = LabeledString('Any major dischord and we all suffer.')
sentence.set_label('tokenization', 'BIEXBIIIEXBIIIIIIEXBIEXBEXBIEXBIIIIES')

# Print the DataPoint
print(sentence)

# Print the string
print(sentence.string)

# print the label
print(sentence.get_labels('tokenization'))

from flair.models.tokenizer_model import FlairTokenizer

# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(character_size, embedding_dim, hidden_dim, num_layers, tagset_size, batch_size)

# do a forward pass and compute the loss for the data point
loss = tokenizer.forward_loss(sentence)

# loss should be a single value tensor 
print(loss)
