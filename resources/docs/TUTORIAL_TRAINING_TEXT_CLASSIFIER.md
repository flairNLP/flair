# Tutorial 4: Tagging your own Models

This tutorial shows you how to train your own NLP models with Flair. 

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_BASICS.md) of this
library and how [embeddings](/resources/docs/TUTORIAL_EMBEDDINGS_OVERVIEW.md) work. 


## Training a Text Classification Model

Training other types of models is very similar to the scripts for training sequence labelers above. For text
classification, use an appropriate corpus and use document-level embeddings instead of word-level embeddings (see
tutorials on both for difference). The rest is exactly the same as before!

The best results in text classification use fine-tuned transformers with `TransformerDocumentEmbeddings` as shown in the
code below:

(If you don't have a big GPU to fine-tune transformers, try `DocumentPoolEmbeddings` or `DocumentRNNEmbeddings` instead; sometimes they work just as well!)

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

Once the model is trained you can load it to predict the class of new sentences. Just call the `predict` method of the
model.

```python
classifier = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')

# create example sentence
sentence = Sentence('Who built the Eiffel Tower ?')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)
```
