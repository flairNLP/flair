# Tutorial 4.1: How Model Training works in Flair

In Flair, all models are trained the same way using the ModelTrainer. This tutorial illustrates 
how the ModelTrainer works and what decisions you have to make to train good models. 


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

This code (1) loads the English universal dependencies dataset as training corpus, (2) create a label dictionary for
universal part-of-speech tags from the corpus, (3) initializes embeddings and (4) runs the trainer for 10 epochs. 

Running this script should produce output that looks like this during training: 

```console
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

```console
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

This report ives us a breakdown of the precision, recall and F1 score of all classes, as well as overall.

Now let's look at the steps in the training script one-by-one: 

### Step 1: Loading the Corpus 

In this example, we use the English Universal Dependencies Dataset to train on. It contains many sentences fully annotated
with both universal and language-specific part-of-speech tags. Running these lines will load and print the corpus: 

```python
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)
```

which should print:
```console
Corpus: 1254 train + 200 dev + 208 test sentences
```

Showing us that our downsampled training data has three splits: a training split of 1254 sentences, a dev split of 200 sentences, and a test split of 208 sentences.

The **Corpus** is a very handy concept in Flair, with lots of helper functions. To learn all that it can do, check out [this tutorial](/resources/docs/TUTORIAL_CORPUS_PREPARED.md).

### Step 2: Creating a Label Dictionary 

Our model needs to predict a set of labels. To determine the label set, run make_label_dictionary on the corpus 
and pass the label type you want to predict. In this example, we pass upos since we want to predict universal 
part-of-speech tags. 

Running these lines will compute and print the label dictionary from the corpus: 

```python
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)
```

which should print:
```console
Dictionary with 18 tags: <unk>, NOUN, PUNCT, VERB, PRON, ADP, DET, AUX, ADJ, PROPN, ADV, CCONJ, PART, SCONJ, NUM, X, SYM, INTJ
```

Showing us that our label dictionary has 18 PoS tags, including one generic tag for all unknown labels.

### Step 3: Initialize the Model

Depending on what you want to do, you need to initialize the appropriate model type. For sequence labeling 
(NER, part-of-speech tagging) you need the SequenceLabeler. For text classification you need to TextClassifier.
For each model type, we are creating dedicated tutorials to better explain what they do.

For this example, we use the SequenceLabeler since we do part-of-speech tagging: 

```python
# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type)
```

Printing it will give you the PyTorch model that is initialized. 

### Step 4: Initialize the Trainer

The ModelTrainer is initialized simply by passing the model and the corpus because that is all it needs.

```python
trainer = ModelTrainer(model, corpus)
```

### Step 5: Train

Once the trainer is initialized, you can call `train` to launch a standard training run. 

```python
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)
```

This will launch a "standard training run" with SGD as optimizer. By default, the learning rate is annealed against the development score: if 
fo 3 epochs there is no improvement on the dev split, the learning rate is halved. If this happens too often, the learning rate will fall below
a minimal threshold and training stops early. 

The max_epochs parameter is set to a small number in this script to make it run fast, but normally you should use a much higher value (150 or 200). 


### Step 6: Predict

Once the model is trained you can use it to predict tags for new sentences. Just call the `predict` method of the model.

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


## Training vs Fine-Tuning

## Other Features

### Plotting Training Curves and Weights

Flair includes a helper method to plot training curves and weights in the neural network. The `ModelTrainer`
automatically generates a `loss.tsv` in the result folder. If you set
`write_weights=True` during training, it will also generate a `weights.txt` file.

After training, simple point the plotter to these files:

```python
# set write_weights to True to write weights
trainer.train('resources/taggers/example-universal-pos',
              ...
write_weights = True,
                ...
)

# visualize
from flair.visual.training_curves import Plotter

plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')
```

This generates PNG plots in the result folder.

### Resuming Training

If you want to stop the training at some point and resume it at a later point, you should train with the parameter
`checkpoint` set to `True`. This will save the model plus training parameters after every epoch. Thus, you can load the
model plus trainer at any later point and continue the training exactly there where you have left.

The example code below shows how to train, stop, and continue training of a `SequenceTagger`. The same can be done
for `TextClassifier`.

```python
from flair.data import Corpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus: Corpus = UD_ENGLISH().downsample(0.1)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 4. initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(hidden_size=128,
                                        embeddings=WordEmbeddings('glove'),
                                        tag_dictionary=label_dict,
                                        tag_type=label_type)

# 5. initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 6. train for 10 epochs with checkpoint=True
path = 'resources/taggers/example-pos'
trainer.train(path,
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10,
              checkpoint=True,
              )

# 7. continue training at later point. Load previously trained model checkpoint, then resume
trained_model = SequenceTagger.load(path + '/checkpoint.pt')

# resume training best model, but this time until epoch 25
trainer.resume(trained_model,
               base_path=path + '-resume',
               max_epochs=25,
               )
```

## Scalability: Training with Large Datasets

Many embeddings in Flair are somewhat costly to produce in terms of runtime and may have large vectors. Examples of this
are Flair- and Transformer-based embeddings. Depending on your setup, you can set options to optimize training time.

### Setting the Mini-Batch Size

The most important is `mini_batch_size`: Set this to higher values if your GPU can handle it to get good speed-ups. However, if
your data set is very small don't set it too high, otherwise there won't be enough learning steps per epoch.

A similar parameter is `mini_batch_chunk_size`: This parameter causes mini-batches to be further split into chunks, causing slow-downs
but better GPU-memory effectiveness. Standard is to set this to None (just don't set it) - only set this if your GPU cannot handle the desired
mini-batch size. Remember that this is the opposite of `mini_batch_size` so this will slow down computation.

### Setting the Storage Mode of Embeddings

Another main parameter you need to set is the `embeddings_storage_mode` in the `train()` method of the `ModelTrainer`. It
can have one of three values:

1. **'none'**: If you set `embeddings_storage_mode='none'`, embeddings do not get stored in memory. Instead they are
   generated on-the-fly in each training mini-batch (during *training*). The main advantage is that this keeps your
   memory requirements low. Always set this if fine-tuning transformers.

2. **'cpu'**: If you set `embeddings_storage_mode='cpu'`, embeddings will get stored in regular memory.

* during *training*: this in many cases speeds things up significantly since embeddings only need to be computed in the
  first epoch, after which they are just retrieved from memory. A disadvantage is that this increases memory
  requirements. Depending on the size of your dataset and your memory setup, this option may not be possible.
* during *inference*: this slows down your inference when used with a GPU as embeddings need to be moved from GPU memory
  to regular memory. The only reason to use this option during inference would be to not only use the predictions but
  also the embeddings after prediction.

3. **'gpu'**: If you set `embeddings_storage_mode='gpu'`, embeddings will get stored in CUDA memory. This will often be
   the fastest one since this eliminates the need to shuffle tensors from CPU to CUDA over and over again. Of course,
   CUDA memory is often limited so large datasets will not fit into CUDA memory. However, if the dataset fits into CUDA
   memory, this option is the fastest one.


## Next

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) to embed your
text.
