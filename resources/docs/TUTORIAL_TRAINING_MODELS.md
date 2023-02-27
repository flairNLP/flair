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

Let's look at these steps one-by-one: 

### Step 1: Loading the Corpus 

In this example, we use the English Universal Dependencies Dataset to train on. It contains many sentences fully annotated
with both universal and language-specific part-of-speech tags. Running these lines will load and print the corpus: 

```python
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)
```

which should print:
```console
...
```

Showing us that our training data has three splits: a training split of X sentences and a test split of X sentences.

The corpus is a very handy object in Flair. To learn all that it can do, check out this tutorial.

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
...
```

Showing us that our label dictionary has ...

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

This will 


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
