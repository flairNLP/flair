# Tutorial 10: Few-Shot and Zero-Shot Classification (TARS)

Task-aware representation of sentences (TARS) was introduced by [Halder et al. (2020)](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf) as a simple and effective 
method for **few-shot and even zero-shot learning for text classification**. This means you can classify
text without (m)any training examples. 
This model is implemented in Flair by the `TARSClassifier` class.
 
In this tutorial, we will show you different ways of using TARS: 

    
## Use Case #1: Classify Text Without Training Data (Zero-Shot)

In some cases, you might not have any training data for the text classification task you want to solve. In this case, 
you can load our default TARS model and do zero-shot prediction. That is, you use the `predict_zero_shot` method
of TARS and give it a list of label names. TARS will then try to match one of these labels to the text.

For instance, say you want to predict whether text is "happy" or "sad" but you have no training data for this. 
Just use TARS with this snippet:

```python
from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence

# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

# 2. Prepare a test sentence
sentence = Sentence("I am so glad you liked it!")

# 3. Define some classes that you want to predict using descriptive names
classes = ["happy", "sad"]

#4. Predict for these classes
tars.predict_zero_shot(sentence, classes)

# Print sentence with predicted labels
print(sentence)
```
The output should look like:
```console
Sentence: "I am so glad you liked it !"   [− Tokens: 8  − Sentence-Labels: {'label': [happy (0.9312)]}]
```

So the label "happy" was chosen for this sentence. 

Try it out with some other labels! Zero-shot prediction will sometimes (*but not always*) work remarkably well. 


## Use Case #2: Classify Text With Very Little Training Data (Few-Shot)
 
While TARS can predict new classes even without training data, it's always better to provide some training examples.
TARS can learn remarkably well from as few as 1 - 10 training examples. 

For instance, assume you want to predict whether a text talks about "food" or "drink". Let's try zero-shot first: 

```python
# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

# 2. Prepare a test sentence
sentence = Sentence("I am so glad you like burritos!")

# 3. Predict zero-shot for classes "food" and "drink", and print results 
tars.predict_zero_shot(sentence, ["food", "drink"])
print(sentence)
```

In this case, zero-shot prediction falsely predicts "drink" for the sentence "I am so glad you like burritos!". 

To improve this, let's first create a small corpus of 4 training and 2 testing examples: 

```python
from flair.data import Corpus
from flair.datasets import SentenceDataset

# training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
train = SentenceDataset(
    [
        Sentence('I eat pizza').add_label('food_or_drink', 'food'),
        Sentence('Hamburgers are great').add_label('food_or_drink', 'food'),
        Sentence('I love drinking tea').add_label('food_or_drink', 'drink'),
        Sentence('Beer is great').add_label('food_or_drink', 'drink')
    ])

# test dataset consisting of two sentences (1 labeled as "food" and 1 labeled as "drink")
test = SentenceDataset(
    [
        Sentence('I ordered pasta').add_label('food_or_drink', 'food'),
        Sentence('There was fresh juice').add_label('food_or_drink', 'drink')
    ])

# make a corpus with train and test split
corpus = Corpus(train=train, test=test)
```

Here, we've named the label class "food_or_drink" with values "food" or "drink". So we want to learn to predict
whether a sentence mentions food or drink. 

Now, let's take the Corpus we created and do few-shot learning with our pre-trained TARS: 

```python
from flair.trainers import ModelTrainer

# 1. load base TARS
tars = TARSClassifier.load('tars-base')

# 2. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task("FOOD_DRINK", label_dictionary=corpus.make_label_dictionary())

# 3. initialize the text classifier trainer with your corpus
trainer = ModelTrainer(tars, corpus)

# 4. train model
trainer.train(base_path='resources/taggers/food_drink', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=1, # small mini-batch size since corpus is tiny
              max_epochs=10, # terminate after 10 epochs
              train_with_dev=True,
              )
```

Done! Let's load the newly trained model and see if it does better: 

```python
# 1. Load few-shot TARS model
tars = TARSClassifier.load('resources/taggers/food_drink/final-model.pt')

# 2. Prepare a test sentence
sentence = Sentence("I am so glad you like burritos")

# 3. Predict for food and drink
tars.predict(sentence)
print(sentence)
```

The model should work much better now! Note that now we just used `predict` instead of `predict_zero_shot`. This 
is because TARS remembers the last task it was trained to do and will do this by default.

Of course, more than 4 training examples works even better. Try it out! 


## Use Case #3: Train Your Own Base TARS Model

Our base TARS model was trained for English using the bert-base-uncased model with 9 text classification
datasets. But for many reasons you might
 want to train your own base TARS model: you may wish to train using more (or different) text classification
datasets, train models for other languages or domains, etc. 

Here's how to do this: 

### How to train with one dataset

Training with one dataset is almost exactly like training a normal text classifier in Flair. The only 
difference is that it sometimes makes sense to rephrase label names into natural language descriptions. 
For instance, the TREC dataset defines labels like "ENTY" that we rephrase to "question about entity".
Better descriptions help TARS learn.

The full training code is then as follows:

```python
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
tars = TARSClassifier(task_name='TREC_6', label_dictionary=corpus.make_label_dictionary())

# 4. initialize the text classifier trainer
trainer = ModelTrainer(tars, corpus)

# 5. start the training
trainer.train(base_path='resources/taggers/trec', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )
```

Done! This trains a classifier for one task that you can now either use as a normal classifier, or as basis for 
few-shot and zero-shot prediction. 


### How to train with multiple datasets

TARS gets better at few-shot and zero-shot prediction if it learns from more than one classification task. 

For instance, lets continue training the model we trained for TREC_6 with the GO_EMOTIONS dataset. The code
again looks very similar, with the difference that we now call `add_and_switch_to_new_task` to make the model
aware that it should train GO_EMOTIONS now instead of TREC_6:

```python
from flair.datasets import GO_EMOTIONS

# 1. Load the trained model
tars = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. load a new flair corpus e.g., GO_EMOTIONS, SENTIMENT_140 etc
new_corpus = GO_EMOTIONS()

# 3. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task( "GO_EMOTIONS",
                                     label_dictionary=new_corpus.make_label_dictionary())

# 4. initialize the text classifier trainer
trainer = ModelTrainer(tars, new_corpus)

# 5. start the training
trainer.train(base_path='resources/taggers/go_emotions', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )
```

At the end of this training, the resulting model can make high quality predictions for 
both TREC_6 and GO_EMOTIONS and is an even better basis for few-shot learning than before.



## Switching between Tasks

TARS can encapsulate the relationship between label names and the text in the underlying 
language model. A single model can be trained on multiple corpora like above. For convenience, it 
internally groups set of labels into different tasks. A user can look up what existing 
tasks a TARS model was trained on, and then switch to one of them as needed.

```python

# 1. Load a pre-trained TARS model
tars = TARSClassifier.load('tars-base')

# 2. Check out what datasets it was trained on
existing_tasks = tars.list_existing_tasks()
print(f"Existing tasks are: {existing_tasks}")

# 3. Switch to a particular task that exists in the above list
tars.switch_to_task("GO_EMOTIONS")

# 4. Prepare a test sentence
sentence = Sentence("I absolutely love this!")
tars.predict(sentence)
print(sentence)
```
The output should look like:
```
Existing tasks are: {'AGNews', 'DBPedia', 'IMDB', 'SST', 'TREC_6', 'NEWS_CATEGORY', 'Amazon', 'Yelp', 'GO_EMOTIONS'}
Sentence: "I absolutely love this !"   [− Tokens: 5  − Sentence-Labels: {'label': [LOVE (0.9708)]}]
```

## Please cite the following paper when using TARS:

```
@inproceedings{halder2020coling,
  title={Task Aware Representation of Sentences for Generic Text Classification},
  author={Halder, Kishaloy and Akbik, Alan and Krapac, Josip and Vollgraf, Roland},
  booktitle = {{COLING} 2020, 28th International Conference on Computational Linguistics},
  year      = {2020}
}
```
