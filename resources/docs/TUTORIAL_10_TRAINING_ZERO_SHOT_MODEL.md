# Tutorial 10: Training a Zero-shot Classifier (TARS)

Task Aware Representation of Sentences (TARS) was introduced by Halder et. al. It formulates the
traditional document level text classification problem as a universal binary text classification
problem. It uses the label names in the classification process, and leverages on the attention mechanism
of transformer based language models. This model is implemented by the `TARSClassifier` class. In this
tutorial, we will show you how to use this class in different scenarios involving training, testing on
unseen datasets.

## Training a TARS Classification Model

Once a TARS classification model is trained on one (or multiple) dataset(s), the final model
can be used in many different ways such as:
* A regular text classifier.
* A Zero-shot predictor on an arbitrary ad-hoc set of labels.
* A Zero-shot predictor on a previously unseen classification dataset. 
* A base model to continue training on other datasets.

```python
import flair
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer


# 1. define label names in natural language since some datasets come with cryptic set of labels
label_name_map = {'ENTY':'question about entity',
                  'DESC':'question about description',
                  'ABBR':'question about abbreviation',
                  'HUM':'question about person',
                  'NUM':'question about number',
                  'LOC':'question about location'
                  }

# 2. get the corpus                  
corpus: Corpus = TREC_6(label_name_map=label_name_map)

# 3. create a TARS classifier
classifier = TARSClassifier(task_name='TREC_6', label_dictionary=corpus.make_label_dictionary())

# 4. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 5. start the training
trainer.train(base_path='resources/taggers/trec', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )


```

## Use Case #1: As a Regular Text Classifier
```python

# 1. Load the trained model
classifier = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. Prepare a test sentence
sentence = flair.data.Sentence("What is the full form of NASA?")
classifier.predict(sentence)
print(sentence)

```
It should output:
```
loading file resources/taggers/trec/best-model.pt
Sentence: "What is the full form of NASA ?"   [− Tokens: 8  − Sentence-Labels: {'label': [question about abbreviation (0.9922)]}]
```

## Use Case #2: As a Zero-shot Predictor on Arbitrary Ad-hoc Set of Labels
```python

# 1. Load the trained model
classifier = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. Prepare a test sentence
sentence = flair.data.Sentence("I am so glad you liked it!")
classifier.predict_zero_shot(sentence, ["happy", "sad"])
print(sentence)

```
It should output:
```
loading file resources/taggers/trec/best-model.pt
Sentence: "I am so glad you liked it !"   [− Tokens: 8  − Sentence-Labels: {'label': [happy (0.9874)]}]
```


## Use Case #3: As a Zero-shot Predictor on a Unseen Task
```python

# 1. Load the trained model
classifier = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. Define a new task
classifier.add_and_switch_to_new_task("sentiment", label_dictionary=["happy", "sad"])

# 3. Prepare a test sentence
sentence = flair.data.Sentence("I am so glad you liked it!") # no need to provide the labels in this way
classifier.predict(sentence)
print(sentence)

```
It should output:
```
loading file resources/taggers/trec/best-model.pt
Sentence: "I am so glad you liked it !"   [− Tokens: 8  − Sentence-Labels: {'label': [happy (0.9874)]}]
```

## Use Case #4: As a Base Model to Continue Training on other Datasets
```python
from flair.datasets import GO_EMOTIONS

# 1. Load the trained model
classifier = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. load a new flair corpus e.g., GO_EMOTIONS, SENTIMENT_140 etc
new_corpus = GO_EMOTIONS()

# 3. make the model aware of the desired set of labels from the new corpus
classifier.add_and_switch_to_new_task( "GO_EMOTIONS",
                                     label_dictionary=new_corpus.make_label_dictionary())

# 4. initialize the text classifier trainer
trainer = ModelTrainer(classifier, new_corpus)

# 5. start the training
trainer.train(base_path='resources/taggers/go_emotions', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )

```
At the end of this training, the resultant model should be able make predictions for both TREC_6 and GO_EMOTIONS tasks with full supervision accuracy.

## Switching between Tasks

TARS can encapsulate the relationship between label names and the text in the underlying 
language model. A single model can be trained on multiple corpora like above. For convenience, it 
internally groups set of labels into different tasks. A user can look up what existing 
tasks a TARS model was trained on, and then switch to one of them as needed.

```python

# 1. Load a pre-trained TARS model
classifier = TARSClassifier.load('resources/taggers/go_emotions/best-model.pt')

# 2. Check out what datasets it was trained on
classifier.list_existing_tasks()

# 3. Switch to a particular task that exists in the above list
classifier.switch_to_task("GO_EMOTIONS")

sentence = flair.data.Sentence("I absolutely love this!")
classifier.predict(sentence)
print(sentence)
```
It should output the following:
```
loading file resources/taggers/go_emotions/best-model.pt
Existing tasks are:
TREC_6
GO_EMOTIONS
Sentence: "I absolutely love this !"   [− Tokens: 5  − Sentence-Labels: {'label': [LOVE (0.9708)]}]
```

## Please cite the following paper when using TARS:

```
@inproceedings{halder2020coling,
  title={Task-Aware Representation of Sentences for Generic Text Classification},
  author={Halder, Kishaloy and Akbik, Alan and Krapac, Josip and Vollgraf, Roland},
  booktitle = {{COLING} 2020, 28th International Conference on Computational Linguistics},
  year      = {2020}
}
```