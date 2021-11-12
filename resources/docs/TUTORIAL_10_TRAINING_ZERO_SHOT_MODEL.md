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
from flair.models import TARSClassifier
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

## Use Case #2: Zero-shot Named Entity Recognition (NER) with TARS

We extend the TARS zero-shot learning approach to sequence labeling and ship a pre-trained model for English NER. Try defining some classes and see if the model can find them: 

```python
from flair.models import TARSTagger
from flair.data import Sentence

# 1. Load zero-shot NER tagger
tars = TARSTagger.load('tars-ner')

# 2. Prepare some test sentences
sentences = [
    Sentence("The Humboldt University of Berlin is situated near the Spree in Berlin, Germany"),
    Sentence("Bayern Munich played against Real Madrid"),
    Sentence("I flew with an Airbus A380 to Peru to pick up my Porsche Cayenne"),
    Sentence("Game of Thrones is my favorite series"),
]

# 3. Define some classes of named entities such as "soccer teams", "TV shows" and "rivers"
labels = ["Soccer Team", "University", "Vehicle", "River", "City", "Country", "Person", "Movie", "TV Show"]
tars.add_and_switch_to_new_task('task 1', labels, label_type='ner')

# 4. Predict for these classes and print results
for sentence in sentences:
    tars.predict(sentence)
    print(sentence.to_tagged_string("ner"))
```

This should print: 

```console
The Humboldt <B-University> University <I-University> of <I-University> Berlin <E-University> is situated near the Spree <S-River> in Berlin <S-City> , Germany <S-Country>

Bayern <B-Soccer Team> Munich <E-Soccer Team> played against Real <B-Soccer Team> Madrid <E-Soccer Team>

I flew with an Airbus <B-Vehicle> A380 <E-Vehicle> to Peru <S-City> to pick up my Porsche <B-Vehicle> Cayenne <E-Vehicle>

Game <B-TV Show> of <I-TV Show> Thrones <E-TV Show> is my favorite series
```


So in these examples, we are finding entity classes such as "TV show" (_Game of Thrones_), "vehicle" (_Airbus A380_ and _Porsche Cayenne_), "soccer team" (_Bayern Munich_ and _Real Madrid_) and "river" (_Spree_), even though the model was never explicitly trained for this. Note that this is ongoing research and the examples are a bit cherry-picked. We expect the zero-shot model to improve quite a bit until the next release.
## Use Case #3: Train a TARS model 

You can also train your own TARS model, either from scratch or by using the provided TARS model as a starting
point. If you chose the latter, you might need only few training data to train a new task.

### How to train with one dataset

Training with one dataset is almost exactly like training any other model in Flair. The only 
difference is that it sometimes makes sense to rephrase label names into natural language descriptions. 
For instance, the TREC dataset defines labels like "ENTY" that we rephrase to "question about entity".
Better descriptions help TARS learn.

The full training code is then as follows:

```python
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

# 1. define label names in natural language since some datasets come with cryptic set of labels
label_name_map = {'ENTY': 'question about entity',
                  'DESC': 'question about description',
                  'ABBR': 'question about abbreviation',
                  'HUM': 'question about person',
                  'NUM': 'question about number',
                  'LOC': 'question about location'
                  }

# 2. get the corpus
corpus: Corpus = TREC_6(label_name_map=label_name_map)

# 3. what label do you want to predict?
label_type = 'question_class'

# 4. make a label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 5. start from our existing TARS base model for English
tars = TARSClassifier.load("tars-base")

# 5a: alternatively, comment out previous line and comment in next line to train a new TARS model from scratch instead
# tars = TARSClassifier(embeddings="bert-base-uncased")

# 6. switch to a new task (TARS can do multiple tasks so you must define one)
tars.add_and_switch_to_new_task(task_name="question classification",
                                label_dictionary=label_dict,
                                label_type=label_type,
                                )

# 7. initialize the text classifier trainer
trainer = ModelTrainer(tars, corpus)

# 8. start the training
trainer.train(base_path='resources/taggers/trec',  # path to store the model artifacts
              learning_rate=0.02,  # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=1,  # terminate after 10 epochs
              )
```

This script starts from the TARS-base model, so a few epochs should be enough. But if you train a new TARS model from scratch instead 
(see step 5a in the code snippet above) you will want to train for 10 or 20 epochs.


### How to train with multiple datasets

TARS gets better at few-shot and zero-shot prediction if it learns from more than one classification task. 

For instance, lets continue training the model we trained for TREC_6 with the GO_EMOTIONS dataset. The code
again looks very similar. Just before you train on the new dataset, be sure to call `add_and_switch_to_new_task`.
This lets the model know that it should train GO_EMOTIONS now instead of TREC_6:

```python
from flair.datasets import GO_EMOTIONS
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

# 1. Load the trained model
tars = TARSClassifier.load('resources/taggers/trec/best-model.pt')

# 2. load a new flair corpus e.g., GO_EMOTIONS, SENTIMENT_140 etc
new_corpus = GO_EMOTIONS()

# 3. define label type
label_type = "emotion"

# 4. make a label dictionary
label_dict = new_corpus.make_label_dictionary(label_type=label_type)

# 5. IMPORTANT: switch to new task
tars.add_and_switch_to_new_task("GO_EMOTIONS",
                                label_dictionary=label_dict,
                                label_type=label_type)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(tars, new_corpus)

# 6. start the training
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
