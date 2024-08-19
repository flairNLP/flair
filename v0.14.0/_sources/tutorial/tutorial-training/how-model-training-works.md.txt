# How model training works in Flair

In this section, we explain the main ideas of model training in Flair.

In particular, we give an introduction to the [`ModelTrainer`](#flair.trainers.ModelTrainer) class, and discuss what decisions you have to make to train good models.

## Example: Training a Part-of-Speech Tagger

As example in this chapter, we train a simple part-of-speech tagger for English. To make the example run fast

- we downsample the training data to 10%
- we use only simple classic word embeddings (gloVe)

Here is the full training code:

```python
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. load the corpus
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embeddings = WordEmbeddings('glove')

# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(model, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)
```

This code (1) loads the English universal dependencies dataset as training corpus, (2) create a label dictionary for universal part-of-speech tags from the corpus, (3) initializes embeddings and (4) runs the trainer for 10 epochs.

Running this script should produce output that looks like this during training:

```
2023-02-27 17:07:38,014 ----------------------------------------------------------------------------------------------------
2023-02-27 17:07:38,016 Model training base path: "resources/taggers/example-upos"
2023-02-27 17:07:38,017 ----------------------------------------------------------------------------------------------------
2023-02-27 17:07:38,020 Device: cuda:0
2023-02-27 17:07:38,022 ----------------------------------------------------------------------------------------------------
2023-02-27 17:07:38,023 Embeddings storage mode: cpu
2023-02-27 17:07:38,025 ----------------------------------------------------------------------------------------------------
2023-02-27 17:07:39,128 epoch 1 - iter 4/40 - loss 3.28409882 - time (sec): 1.10 - samples/sec: 2611.84 - lr: 0.100000
2023-02-27 17:07:39,474 epoch 1 - iter 8/40 - loss 3.13510367 - time (sec): 1.45 - samples/sec: 3143.21 - lr: 0.100000
2023-02-27 17:07:39,910 epoch 1 - iter 12/40 - loss 3.02619775 - time (sec): 1.88 - samples/sec: 3434.39 - lr: 0.100000
2023-02-27 17:07:40,167 epoch 1 - iter 16/40 - loss 2.95288554 - time (sec): 2.14 - samples/sec: 3783.76 - lr: 0.100000
2023-02-27 17:07:40,504 epoch 1 - iter 20/40 - loss 2.86820018 - time (sec): 2.48 - samples/sec: 4171.22 - lr: 0.100000
2023-02-27 17:07:40,843 epoch 1 - iter 24/40 - loss 2.80507526 - time (sec): 2.82 - samples/sec: 4557.72 - lr: 0.100000
2023-02-27 17:07:41,118 epoch 1 - iter 28/40 - loss 2.74217397 - time (sec): 3.09 - samples/sec: 4878.00 - lr: 0.100000
2023-02-27 17:07:41,420 epoch 1 - iter 32/40 - loss 2.69161746 - time (sec): 3.39 - samples/sec: 5072.93 - lr: 0.100000
2023-02-27 17:07:41,705 epoch 1 - iter 36/40 - loss 2.63837577 - time (sec): 3.68 - samples/sec: 5260.02 - lr: 0.100000
2023-02-27 17:07:41,972 epoch 1 - iter 40/40 - loss 2.58915523 - time (sec): 3.95 - samples/sec: 5394.33 - lr: 0.100000
2023-02-27 17:07:41,975 ----------------------------------------------------------------------------------------------------
2023-02-27 17:07:41,977 EPOCH 1 done: loss 2.5892 - lr 0.100000
2023-02-27 17:07:42,567 DEV : loss 2.009714126586914 - f1-score (micro avg)  0.41
2023-02-27 17:07:42,579 BAD EPOCHS (no improvement): 0
```

The output monitors the loss over the epochs. At the end of each epoch, the development score is computed and printed.

And a **final evaluation report** gets printed in the end:

```
Results:
- F-score (micro) 0.7732
- F-score (macro) 0.6329
- Accuracy 0.7732

By class:
              precision    recall  f1-score   support

        NOUN     0.7199    0.7199    0.7199       407
       PUNCT     0.9263    0.9843    0.9544       319
        VERB     0.7521    0.6938    0.7218       258
        PRON     0.7782    0.9300    0.8474       200
         ADP     0.8559    0.9515    0.9011       206
       PROPN     0.6585    0.6398    0.6490       211
         ADJ     0.5654    0.6914    0.6221       175
         DET     0.9572    0.8995    0.9275       199
         AUX     0.8609    0.8784    0.8696       148
         ADV     0.5052    0.5000    0.5026        98
       CCONJ     0.9833    0.9077    0.9440        65
         NUM     0.5435    0.3289    0.4098        76
        PART     0.9091    0.7143    0.8000        56
       SCONJ     0.7083    0.5667    0.6296        30
         SYM     0.3333    0.2143    0.2609        14
           X     0.0000    0.0000    0.0000        15
        INTJ     0.0000    0.0000    0.0000        14

    accuracy                         0.7732      2491
   macro avg     0.6504    0.6247    0.6329      2491
weighted avg     0.7635    0.7732    0.7655      2491
```

This report gives us a breakdown of the precision, recall and F1 score of all classes, as well as overall.

Congrats, you just trained your first model!


## Step-by-step walkthrough

The above code showed you how to train a part-of-speech tagger.

Now let's individually look at each of the main steps in the above script:

### Step 1: Load a Corpus

The first thing you need is data to train and evaluate your model on.

In Flair, training is done using the [`Corpus`](#flair.data.Corpus) object that holds three "splits": a `train`, a `dev` and a `test` split.

```{note}

Splitting your data into three splits is standard procedure in machine learning: the `train` split is used to train the model while the `dev` split is used for model selection and early stopping. The `test` split is used only for the final evaluation.
```

In this example, we use the [English Universal Dependencies](https://universaldependencies.org/treebanks/en_ewt/index.html) dataset to train on. It contains many sentences fully annotated with both universal and language-specific part-of-speech tags. Running these lines will load and print the corpus:

```python
from flair.datasets import UD_ENGLISH

# 1. load the corpus
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)
```

which should print:

```
Corpus: 1254 train + 200 dev + 208 test sentences
```

Showing us that our downsampled training data has three splits: a training split of 1254 sentences, a dev split of 200 sentences, and a test split of 208 sentences.

```{note}
The [`Corpus`](#flair.data.Corpus) object has a number of very handy helper functions that let you manipulate the data and compute statistics. For instance, in the code above we called [`Corpus.downsample(0.1)`](#flair.data.Corpus.downsample) to downsample the corpus to 10% of its original size. To learn about more helper functions, check out the [corpus tutorial](how-to-load-prepared-dataset.md).
```

### Step 2: Choose the label type

After you load the corpus, you need to decide which label type to predict.

We choose the label type **'upos'**, since we want to predict universal part-of-speech tags in this example.

```python
# 2. what label do we want to predict?
label_type = 'upos'
```

```{note}
You might ask: why is specifying the `label_type` even necessary? Well, some corpora have more than one label type. The English UD treebank for instance has both universal PoS tags ('upos') and regular PoS tags ('pos'), plus many other layers of annotation. A tagger is normally trained to predict just type of annotation.

This means that you need to know which label types a specific corpus has labels for, and choose one of them.
```


### Step 3: Creating a label dictionary

Our model needs to predict a set of labels. To determine the label set, run [`Corpus.make_label_dictionary()`](#flair.data.Corpus.make_label_dictionary) on the corpus and pass the label type you want to predict. In this example, we pass **'upos'** since we want to predict universal part-of-speech tags.

Running these lines will compute and print the label dictionary from the corpus:

```python
# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)
```

which should print:

```
Dictionary with 18 tags: <unk>, NOUN, PUNCT, VERB, PRON, ADP, DET, AUX, ADJ, PROPN, ADV, CCONJ, PART, SCONJ, NUM, X, SYM, INTJ
```

Showing us that our label dictionary has 18 PoS tags, including one generic tag (`<unk>`) for all unknown labels.

### Step 4: Initialize embeddings

All models in Flair require you to choose embeddings. In most cases, you'll want transformer embeddings. Choosing the right embeddings and parameters is crucial in order to train good models.

In our example, we use simple GloVe embeddings:


```python
# 4. initialize embeddings
embeddings = WordEmbeddings('glove')
```

But this is only to make the example code run fast. We generally advise to use transformer-based embeddings instead.

### Step 5: Initialize the Model

Depending on what you want to do, you need to initialize the appropriate model type.

For this example, we use the [`SequenceTagger`](#flair.models.SequenceTagger) since we do part-of-speech tagging:

```python
# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256,
                       embeddings=embeddings,
                       tag_dictionary=label_dict,
                       tag_type=label_type)
```

Printing it will give you the PyTorch model that is initialized.

```{note}
Depending on the task, you need a different model type: For sequence labeling (NER, part-of-speech tagging) you need the [`SequenceTagger`](#flair.models.SequenceTagger). For text classification you need the [`TextClassifier`](#flair.models.TextClassifier).

For each model type, we are creating dedicated tutorials to better explain what they do.
```

### Step 6: Initialize the Trainer

The [`ModelTrainer`](#flair.trainers.ModelTrainer) is initialized simply by passing the model and the corpus because that is all it needs.

```python
# 6. initialize trainer
trainer = ModelTrainer(model, corpus)
```

### Step 7: Train

Once the trainer is initialized, you can call [`ModelTrainer.train()`](#flair.trainers.ModelTrainer.train) to launch a standard training run.

```python
# 7. start training
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)
```

This will launch a "standard training run" with SGD as optimizer. By default, the learning rate is annealed against the development score: if for 3 epochs there is no improvement on the dev split, the learning rate is halved. If this happens too often, the learning rate will fall below a minimal threshold and training stops early.

The max_epochs parameter is set to a small number in this script to make it run fast, but normally you should use a much higher value (150 or 200).

```{note}
There are two main mechanisms to train a model in Flair. (1) The "classic" workflow (SGD with annealing) is invoked as above using the [`ModelTrainer.train()`](#flair.trainers.ModelTrainer.train) method. (2) The current state-of-the-art based on fine-tuning (AdamW with Linear Learning Rate Schedulle) is invoked using the [`ModelTrainer.fine_tune()`](#flair.trainers.ModelTrainer.fine_tune) method. In most cases, you will want to use the latter.
```

### Step 8: Predict

Once the model is trained you can use it to predict tags for new sentences. Just call the [`.predict()`](#flair.nn.Classifier.predict) method of the model.

```python
# load the model you trained
model = SequenceTagger.load('resources/taggers/example-upos/final-model.pt')

# create example sentence
sentence = Sentence('I love Berlin')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
```

If the model works well, it will correctly tag 'love' as a verb in this example.

## Summary

This tutorial gave you a general overview of the main steps to train a model:

-    load a corpus
-    choose a label type
-    create a label dictionary
-    choose embeddings
-    initialize model
-    initialize trainer
-    train


