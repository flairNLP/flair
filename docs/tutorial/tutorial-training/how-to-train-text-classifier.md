# Train a text classifier

This tutorial shows you how to train your own text classifier models with Flair. For instance, you 
could train your own sentiment analysis model, or offensive language detection model.


## Training a text classification model with transformers

For text classification, you reach state-of-the-art scores by fine-tuning a transformer. 

Training a model is easy: load the appropriate corpus, make a label dictionary, then fine-tune a [`TextClassifier`](#flair.models.TextClassifier)
model using the [`ModelTrainer.fine_tune()`](#flair.trainers.ModelTrainer.fine_tune) method. See the example script below:

```python
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus: Corpus = TREC_6()

# 2. what label do we want to predict?
label_type = 'question_class'

# 3. create the label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 4. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(classifier, corpus)

# 7. run training with fine-tuning
trainer.fine_tune('resources/taggers/question-classification-with-transformer',
                  learning_rate=5.0e-5,
                  mini_batch_size=4,
                  max_epochs=10,
                  )
```

Once the model is trained you can load it to predict the class of new sentences. Just call the [`predict`](#flair.nn.DefaultClassifier.predict) method of the model.

```python
classifier = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')

# create example sentence
sentence = Sentence('Who built the Eiffel Tower ?')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)
```

