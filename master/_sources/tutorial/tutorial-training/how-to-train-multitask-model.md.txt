# Train a Multitask Model

In some cases, you might want to train a single model that can complete multiple tasks. For instance, you might want to 
train a model that can do both part-of-speech tagging and syntactic chunking. Or a model that can predict both entities
and relations. 

In such cases, you typically have a single language model as backbone and multiple prediction heads with task-specific 
prediction logic. The potential advantage is twofold: 

1. Instead of having two separate language models for two tasks, you have a single model, making it more compact if you keep it in memory.
2. The language model will simultaneously learn from the training data of both tasks. This may result in higher accuracy for both tasks if they are semantically close. (In practice however, such effects are rarely observed.) 


## Example 1: Two token-level tasks

Our first multitask example is a single model that predicts part-of-speech tags and syntactic chunks. Both tasks are
token-level prediction tasks that are syntactic and therefore closely related.

The following script loads a single embedding for both tasks, but loads two separate training corpora. From these separate
training corpora, it creates two label dictionaries and instantiates two prediction models (both [TokenClassifier](#flair.models.TokenClassifier)):

```python
from flair.embeddings import TransformerWordEmbeddings
from flair.datasets import UD_ENGLISH, CONLL_2000
from flair.models import TokenClassifier
from flair.trainers import ModelTrainer
from flair.nn.multitask import make_multitask_model_and_corpus

# --- Embeddings that are shared by both models --- #
shared_embedding = TransformerWordEmbeddings("distilbert-base-uncased",
                                             fine_tune=True)

# --- Task 1: Part-of-Speech tagging --- #
corpus_1 = UD_ENGLISH().downsample(0.1)

model_1 = TokenClassifier(shared_embedding,
                         label_dictionary=corpus_1.make_label_dictionary("pos"),
                         label_type="pos")

# -- Task 2: Syntactic Chunking -- #
corpus_2 = CONLL_2000().downsample(0.1)

model_2 = TokenClassifier(shared_embedding,
                          label_dictionary=corpus_2.make_label_dictionary("np"),
                          label_type="np",
                          )

# -- Define mapping (which tagger should train on which model) -- #
multitask_model, multicorpus = make_multitask_model_and_corpus(
    [
        (model_1, corpus_1),
        (model_2, corpus_2),
    ]
)

# -- Create model trainer and train -- #
trainer = ModelTrainer(multitask_model, multicorpus)
trainer.fine_tune(f"resources/taggers/multitask_test")
```

The key is the function `make_multitask_model_and_corpus`, which takes these individual models and corpora and creates a 
single multitask model and corpus out of them. These are then passed to the model trainer as usual. 

When you run this script, it should print a training log like always, just with the difference that at the end of each epoch,
both tasks are evaluated on the dev split: 

```
2025-03-26 22:29:16,187 ----------------------------------------------------------------------------------------------------
2025-03-26 22:29:16,187 EPOCH 1 done: loss 1.3350 - lr: 0.000049
2025-03-26 22:29:16,525 Task_0 - TokenClassifier - loss: 0.28858715295791626 - f1-score (micro avg)  0.9292
2025-03-26 22:29:16,776 Task_1 - TokenClassifier - loss: 7.221250534057617 - f1-score (micro avg)  0.9077
2025-03-26 22:29:16,777 DEV : loss 3.7549188137054443 - f1-score (micro avg)  0.9184
2025-03-26 22:29:16,783 ----------------------------------------------------------------------------------------------------
```

### Prediction

A trained model can be loaded and used for prediction like any other model. However, it will make predictions for all 
tasks it was trained with: 

```python
from flair.data import Sentence
from flair.models import MultitaskModel

# load the trained multitask model
model = MultitaskModel.load("resources/taggers/multitask_test/final-model.pt")

# create example sentence
sentence = Sentence("Mr Smith loves New York")

# predict (this triggers prediction of all tasks in the multitask model)
model.predict(sentence)

# print the sentence with POS and chunk tags
print(sentence)

# inspect the POS tags only
print("\nPart of speech tags are: ")
for label in sentence.get_labels('pos'):
    print(f'"{label.data_point.text}" {label.value}')

# inspect the chunks only
print("\nChunks are: ")
for label in sentence.get_labels('np'):
    print(f'"{label.data_point.text}" {label.value}')
```

This will print: 

```
Sentence[5]: "Mr Smith loves New York" → ["Mr"/NNP, "Mr Smith"/NP, "Smith"/NNP, "loves"/VBZ, "loves"/VP, "New"/NNP, "New York"/NP, "York"/NNP]

Part of speech tags are: 
"Mr" NNP
"Smith" NNP
"loves" VBZ
"New" NNP
"York" NNP

Chunks are: 
"Mr Smith" NP
"loves" VP
"New York" NP
```


## Example 2: A token and a document-level task

In some cases, you may want to train a multitask model using [TransformerWordEmbeddings](#flair.embeddings.transformer.TransformerWordEmbeddings) (token-level embeddings) 
and [TransformerDocumentEmbeddings](#flair.embeddings.transformer.TransformerDocumentEmbeddings) (text-level embeddings). For instance, you may want to train a model that can both
detect topics and entities in online news articles. 

The code is similar to example 1, but you need more general [TransformerEmbeddings](#flair.embeddings.transformer.TransformerEmbeddings) that can produce both token- and text-level
embeddings. You also need two different model classes: A [TextClassifier](#flair.models.TextClassifier) for predicting topics and a [TokenClassifier](#flair.models.TokenClassifier) for
prediction NER tags: 

```python
from flair.embeddings import TransformerEmbeddings
from flair.datasets import AGNEWS, CLEANCONLL
from flair.models import TokenClassifier, TextClassifier
from flair.trainers import ModelTrainer
from flair.nn.multitask import make_multitask_model_and_corpus

# --- Embeddings that are shared by both models --- #
# use a transformer that can do both: sentence-embedding and word-embedding
shared_embedding = TransformerEmbeddings("distilbert-base-uncased",
                                         fine_tune=True,
                                         is_token_embedding=True,
                                         is_document_embedding=True)

# --- Task 1: Newswire topics, use a TextClassifier for this task --- #
corpus_1 = AGNEWS().downsample(0.01)

model_1 = TextClassifier(shared_embedding,
                         label_dictionary=corpus_1.make_label_dictionary("topic"),
                         label_type="topic")

# -- Task 2: Named entities on newswire data, use a TokenClassifier for this task --- #
corpus_2 = CLEANCONLL().downsample(0.1)

model_2 = TokenClassifier(shared_embedding,
                          label_dictionary=corpus_2.make_label_dictionary("ner"),
                          label_type="ner",
                          )

# -- Define mapping (which tagger should train on which model) -- #
multitask_model, multicorpus = make_multitask_model_and_corpus(
    [
        (model_1, corpus_1),
        (model_2, corpus_2),
    ]
)

# -- Create model trainer and train -- #
trainer = ModelTrainer(multitask_model, multicorpus)
trainer.fine_tune(f"resources/taggers/multitask_news_data")
```


### Prediction

Similar to example 1, you can load and predict tags for both classes with a single predict call: 

```python
from flair.data import Sentence
from flair.models import MultitaskModel

# load the trained multitask model
model = MultitaskModel.load("resources/taggers/multitask_news_data/final-model.pt")

# create example sentence
sentence = Sentence("IBM made a lot of profit.")

model.predict(sentence)

# print the sentence with predicted topic and entity tag
print(sentence)
```

This prints: 

```
Sentence[7]: "IBM made a lot of profit." → Business (0.8645) → ["IBM"/ORG]
```

Showing that the topic "Business" and the entity "IBM" were detected in this sentence.