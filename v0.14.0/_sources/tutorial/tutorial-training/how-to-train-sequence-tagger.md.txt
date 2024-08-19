# Train a sequence tagger

Sequence labeling models are used to model problems such as named entity recognition (NER) and
part-of-speech (PoS) tagging.

This tutorial section show you how to train state-of-the-art NER models and other taggers in Flair.

## Training a named entity recognition (NER) model with transformers

For a state-of-the-art NER system you should fine-tune transformer embeddings, and use full document context
(see our [FLERT](https://arxiv.org/abs/2011.06993) paper for details). 

Use the following script:

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
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
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

As you can see, we use [`TransformerWordEmbeddings`](#flair.embeddings.token.TransformerWordEmbeddings) based on 'xlm-roberta-large' embeddings. We enable fine-tuning and set `use_context` to True. 
We also deactivate the RNN, CRF and reprojection in the [`SequenceTagger`](#flair.models.SequenceTagger). This is because the 
transformer is so powerful that it does not need these components. We then fine-tune the model with a very small
learning rate on the corpus.

This will give you state-of-the-art numbers similar to the ones reported
in [Schweter and Akbik (2021)](https://arxiv.org/abs/2011.06993). 


## Training a named entity recognition (NER) model with Flair embeddings

Alternatively to fine-tuning a very large transformer, you can use a classic training setup without fine-tuning.
In the classic setup, you learn a LSTM-CRF on top of frozen embeddings. We typically use a 'stack' that combines
Flair and GloVe embeddings:

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
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
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
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/sota-ner-flair',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```

This will give you state-of-the-art numbers similar to the ones reported in [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf).
The numbers are not quite as high as fine-tuning transformers, but it requires less GPU memory and depending on your
setup may run faster in the end. 


## Training a part-of-speech tagger

If you want to train a part-of-speech model instead of NER, simply exchange the corpus and the label type: 

```python
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = UD_ENGLISH()
print(corpus)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
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
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32)
```

This script will give you the state-of-the-art accuracy reported in [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf).

## Multi-dataset training

Now, let us train a single model that can PoS tag text in both English and German. To do this, we load both the English
and German UD corpora and create a [`MultiCorpus`](#flair.data.MultiCorpus) object. We also use the new multilingual Flair embeddings for this task.

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


