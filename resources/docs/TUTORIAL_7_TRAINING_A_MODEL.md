# Tutorial 7: Training a Model

This part of the tutorial shows how you can train your own sequence labelling and text classification models using
state-of-the-art word embeddings.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) work (ideally, you also know
how [Flair embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md) work). You should also know how
to [load a corpus](/resources/docs/TUTORIAL_6_CORPUS.md).

## Training a Part-of-Speech Tagging Model

Here is example code for a small part-of-speech tagger model trained over UD_ENGLISH (English universal dependency
treebank) data, using simple GloVe embeddings. In this example, we downsample the data to 10% of the original data to
make it run faster, but normally you should train over the full dataset:

```python
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)
```

Alternatively, try using a stacked embedding with FlairEmbeddings and GloVe, over the full data, for 150 epochs. This
will give you the state-of-the-art accuracy reported in [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf).

Once the model is trained you can use it to predict tags for new sentences. Just call the `predict` method of the model.

```python
# load the model you trained
model = SequenceTagger.load('resources/taggers/example-pos/final-model.pt')

# create example sentence
sentence = Sentence('I love Berlin')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
```

If the model works well, it will correctly tag 'love' as a verb in this example.

## Training a Named Entity Recognition (NER) Model with Flair Embeddings

To train a sequence labeling model for NER, just minor modifications to the above script are necessary. Load an NER
corpus like CONLL_03 (requires manual download of data - or use
a [different NER corpus](/resources/docs/TUTORIAL_6_CORPUS.md#datasets-included-in-flair)), change the `label_type` to '
ner' and and use a `StackedEmbedding` consisting of GloVe and Flair:

```python
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = CONLL_03()
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/sota-ner-flair',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```

This will give you state-of-the-art numbers similar to the ones reported
in [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf).

## Training a Named Entity Recognition (NER) Model with Transformers

You can get **even better numbers** if you use transformers as embeddings, fine-tune them and use full document context
(see our [FLERT](https://arxiv.org/abs/2011.06993) paper for details). It's state-of-the-art but much slower than the
above model.

Change the script to use transformer embeddings and change the training routine to fine-tune with AdamW optimizer and a
tiny learning rate instead of SGD:

```python
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = CONLL_03()
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
trainer.fine_tune('resources/taggers/sota-ner-flert',
                  learning_rate=5.0e-6,
                  mini_batch_size=4,
                  mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                  )
```

This will give you state-of-the-art numbers similar to the ones reported
in [Schweter and Akbik (2021)](https://arxiv.org/abs/2011.06993).

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

## Multi-Dataset Training

Now, let us train a single model that can PoS tag text in both English and German. To do this, we load both the English
and German UD corpora and create a MultiCorpus object. We also use the new multilingual Flair embeddings for this task.

All the rest is same as before, e.g.:

```python
from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpora - English and German UD
corpus = MultiCorpus([UD_ENGLISH(), UD_GERMAN()]).downsample(0.1)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embedding_types = [

    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-universal-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              )
```

This gives you a multilingual model. Try experimenting with more languages!

## Plotting Training Curves and Weights

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

## Resuming Training

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

If you don't have training data (or only very little), our TARS approach might be best for you. Check out the TARS
tutorial on [few-shot and zero-shot classification](/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)).

Alternatively, you can look into [training your own embeddings](/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).

