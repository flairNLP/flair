# Tutorial 6: Training your own Character LM Embeddings

Character LM Embeddings are the secret sauce in Flair, allowing us to achieve state-of-the-art accuracies across a range of NLP tasks.
This tutorial shows you how to train your own CharLM embeddings, which may come in handy if you want to apply Flair to new languages or domains.


## Preparing a Text Corpus

Language models are trained with plain text. In the case of character LMs, we train them to predict the next character in a sequence of characters.
To train your own model, you first need to identify a suitably large corpus. In our eperiments, we used corpora that have about 1 billion words.

You need to split your corpus into train, validation and test portions.
Our trainer class assumes that there is a folder for the corpus in which there is a 'test.txt' and a 'valid.txt' with test and validation data.
Importantly, there is also a folder called 'train' that contains the training data in splits.
For instance, the billion word corpus is split into 100 parts.
The splits are necessary if all the data does not fit into memory, in which case the trainer randomly iterates through all splits.

So, the folder structure must look like this:

```
corpus/
corpus/train/
corpus/train/train_split_1
corpus/train/train_split_2
corpus/train/...
corpus/train/train_split_X
corpus/test.txt
corpus/valid.txt
```


## Training the Language Model

Once you have this folder structure, simply point the `LanguageModelTrainer` class to it to start learning a model.

```python
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

# get your corpus, process forward and at the character level
corpus = TextCorpus('/path/to/your/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=128,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/taggers/language_model',
              sequence_length=10,
              mini_batch_size=10,
              max_epochs=10)
```

The parameters in this script are very small. We got good results with a hidden size of 1024 or 2048, a sequence length of 250, and a mini-batch size of 100.
Depending on your resources, you can try training large models, but beware that you need a very powerful GPU and a lot of time to train a model (we train for > 1 week).



## Using the LM as Embeddings

Once you have the LM trained, using it as embeddings is easy. Just load the model into the `CharLMEmbeddings` class and use as you would any other embedding in Flair:

```python
sentence = Sentence('I love Berlin')

# init embeddings from your trained LM
char_lm_embeddings = CharLMEmbeddings('resources/taggers/language_model/best-lm.pt')

# embed sentence
char_lm_embeddings.embed(sentence)
```

Done!


## Parameters

You might to play around with some of the learning parameters in the `LanguageModelTrainer`.
For instance, we generally find that an initial learning rate of 20, and an annealing factor of 4 is pretty good for most corpora.
You might also want to modify the 'patience' value of the learning rate scheduler. We currently have it at 25, meaning that if the training loss does not improve for 25 splits, it decreases the learning rate.




## Consider Contributing your LM

If you train a good LM for a language or domain we don't yet have in Flair, consider contacting us! We would be happy to integrate more LMs into the library so that other people can use them!



