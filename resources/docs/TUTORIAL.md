# Tutorial

Let's look into some core functionality to understand the library better. A [Jupyter notebook](/tutorial.ipynb) version of this tutorial
is available, too.

## NLP base types

There are two types of objects that are central to this library, namely the `Sentence` and `Token` objects. A `Sentence` 
holds a textual sentence and is essentially a list of `Token`.

Let's start by making a `Sentence` object for an example sentence.

```python
# The sentence objects holds a sentence that we may want to embed
from flair.data import Sentence

# Make a sentence object by passing a whitespace tokenized string
sentence = Sentence('The grass is green .')

# Print the object to see what's in there
print(sentence)
```

This should print: 

```console
Sentence: "The grass is green ." - 5 Tokens
```

The print-out tells us that the sentence consists of 5 tokens. 
You can access the tokens of a sentence via their token id:

```python
print(sentence[4])
```

which should print 

```console
Token: 4 green
```

This print-out includes the token id (4) and the lexical value of the token ("green"). You can also iterate over all 
tokens in a sentence.

```python
for token in sentence:
    print(token) 
```

This should print: 

```console
Token: 1 The
Token: 2 grass
Token: 3 is
Token: 4 green
Token: 5 .
```

A Token has fields for linguistic annotation, such as lemmas, part-of-speech tags or named entity tags. You can 
add a tag by specifying the tag type and the tag value. In this example, we're adding an NER tag of type 'color' to 
the word 'green'. This means that we've tagged this word as an entity of type color.

```python
# add a tag to a word in the sentence
sentence[4].add_tag('ner', 'color')

# print the sentence with all tags of this type
print(sentence.to_ner_string())
```

This should print: 

```console
The grass is green <color> .
```


## Tagging with Pre-Trained Models

Now, lets use a pre-trained model for named entity recognition (NER). 
This model was trained over the English CoNLL-03 task and can recognize 4 different entity
types.

```python
from flair.tagging_model import SequenceTaggerLSTM

tagger = SequenceTaggerLSTM.load('ner')
```
All you need to do is use the `predict()` method of the tagger on a sentence. This will add predicted tags to the tokens
in the sentence. Lets use a sentence with two named
entities: 

```python
sentence = Sentence('George Washington went to Washington .')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tag_string())
```

This should print: 
```console
George <B-PER> Washington <E-PER> went <O> to <O> Washington <S-LOC> . <O>
```

You chose which pre-trained model you load by passing the appropriate 
string you pass to the `load()` method of the `SequenceTaggerLSTM` class. Currently, the following pre-trained models
are provided (more coming): 
 
| ID | Task + Training Dataset | Accuracy | 
| -------------    | ------------- | ------------- |
| 'ner' | Conll-03 Named Entity Recognition (English)   |  **93.17** (F1) |
| 'chunk' | Conll-2000 Syntactic Chunking (English)     |  **96.74** (F1) |
| 'pos' | Ontonotes Part-of-Speech Tagging (English)    |  **98.06** (Accuracy) |

So, if you want to use a `SequenceTaggerLSTM` that performs PoS tagging, instantiate the tagger as follows:

```python
tagger = SequenceTaggerLSTM.load('pos')
```


## Embeddings

We provide a set of classes with which you can embed the words in sentences in various ways. Note that all embedding 
classes inherit from the `TextEmbeddings` class and implement the `embed()` method which you need to call 
to embed your text. This means that for most users of Flair, the complexity of different embeddings remains hidden 
behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are pytorch vectors, so they can be immediately used for training and 
fine-tuning.

### Classic Word Embeddings

Classic word embeddings are static and word-level, meaning that each distinc word gets exactly one pre-computed 
embedding. Most embeddings fall under this class, including the popular GloVe or Komnios embeddings. 

Simply instantiate the WordEmbeddings class and pass a string identifier of the embedding you wish to load. So, if 
you want to use GloVe embeddings, pass the string 'glove' to the constructor: 

```python
# all embeddings inherit from the TextEmbeddings class. Init a simple glove embedding.
from flair.embeddings import WordEmbeddings
glove_embedding = WordEmbeddings('glove')
```
Now, create an example sentence and call the embedding's `embed()` method. You always pass a list of sentences to 
this method since some embedding types make use of batching to increase speed. So if you only have one sentence, 
pass a list containing only one sentence:

```python
# embed a sentence using glove.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
glove_embedding.embed(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

This prints out the tokens and their embeddings. GloVe embeddings are pytorch vectors of dimensionality 100.

You choose which pre-trained embeddings you load by passing the appropriate 
string you pass to the constructor of the `WordEmbeddings` class. Currently, the following static embeddings
are provided (more coming): 
 
| ID | Embedding | 
| -------------  | ------------- |
| 'glove' | GloVe embeddings |
| 'extvec' | Komnios embeddings |
| 'ft-crawl' | FastText embeddings |
| 'ft-german' | German FastText embeddings |

So, if you want to load German FastText embeddings, instantiate the method as follows:

```python
german_embedding = WordEmbeddings('ft-german')
```

### Contextual String Embeddings


Contextual string embeddings are [powerful embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and
thus fundamentally model words as sequences of characters. And (2) they are **contextualized** by their
surrounding text, meaning that the *same word will have different embeddings depending on its
contextual use*.

With Flair, you can use these embeddings simply by instantiating the appropriate embedding class, same as before:

```python

# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import CharLMEmbeddings
charlm_embedding_forward = CharLMEmbeddings('news-forward')

# embed a sentence using CharLM.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
charlm_embedding_forward.embed(sentences=[sentence])
```

You choose which embeddings you load by passing the appropriate 
string you pass to the constructor of the `CharLMEmbeddings` class. Currently, the following contextual string
 embeddings
are provided (more coming): 
 
| ID | Language | Embedding | 
| -------------     | ------------- | ------------- |
| 'news-forward'    | English | Forward LM embeddings over 1 billion word corpus |
| 'news-backward'   | English | Backward LM embeddings over 1 billion word corpus |
| 'mix-forward'     | English | Forward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'mix-backward'    | English | Backward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'german-forward'  | German  | Forward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'german-backward' | German  | Backward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |

So, if you want to load embeddings from the English news backward LM model, instantiate the method as follows:

```python
charlm_embedding_backward = CharLMEmbeddings('news-backward')
```


### Character Embeddings

Some embeddings - such as character-features - are not pre-trained but rather trained on the downstream task. Normally
this requires you to implement a [hierarchical embedding architecture](http://neuroner.com/NeuroNERengine_with_caption_no_figure.png). 

With Flair, you need not worry about such things. Just choose the appropriate
embedding class and character features will then automatically train during downstream task training. 

```python
# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import CharacterEmbeddings
embedder = CharacterEmbeddings()

# embed a sentence using CharLM.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
embedder.embed(sentences=[sentence])
```

## Stacked Embeddings

Stacked embeddings are one of the most important concepts of this library. You can use them to combine different embeddings
together, for instance if you want to use both traditional embeddings together with contextual sting embeddings. 
Stacked embeddings allow you to mix and match. We find that a combination of embeddings often gives best results. 

All you need to do is use the `StackedEmbeddings` class and instantiate it by passing a list of embeddings that you wish 
to combine. For instance, lets combine classic GloVe embeddings with embeddings from a forward and backward 
character language model.

First, instantiate the three embeddings you wish to combine: 

```python
# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import WordEmbeddings, CharLMEmbeddings

# init GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init CharLM embedding
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these three embeddings.

```python
# now create the StackedEmbedding object that combines all embeddings
from flair.embeddings import StackedEmbeddings
stacked_embeddings = StackedEmbeddings(embeddings=[glove_embedding, charlm_embedding_forward, charlm_embedding_backward])
```

That's it! Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
# just embed a sentence using the StackedEmbedding as you would with any single embedding.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
stacked_embeddings.embed(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

Words are now embedding using a concatenation of three different embeddings. This means that the resulting embedding
vector is still a single Pytorch vector. 

## Reading an Evaluation Dataset

Flair provides helper 
methods to read common NLP datasets, such as the CoNLL-03 and CoNLL-2000 evaluation datasets, and the
CoNLL-U format. These might be interesting to you if you want to train your own sequence labelers. 

All helper methods for reading data are bundled in the `NLPTaskDataFetcher` class. One option for you is to follow 
the instructions for putting the training data in the appropriate folder structure, and use the prepared functions. 
For instance, if you want to use the CoNLL-03 data, get it from the task Web site 
and place train, test and dev data in `/resources/tasks/conll_03/` as follows: 

```
/resources/tasks/conll_03/eng.testa
/resources/tasks/conll_03/eng.testb
/resources/tasks/conll_03/eng.train
```

This allows the `NLPTaskDataFetcher` class to read the data into our data structures. Use the `NLPTask` enum to select 
the dataset, as follows: 

```python
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
```

This gives you a `TaggedCorpus` object that contains the data. 

However, this only works if the relative folder structure perfectly matches the presets. If not - or you are using 
a different dataset, you can still use the inbuilt functions to read different CoNLL formats:

```python
# use your own data path
data_folder = 'path/to/your/data'

# get training, test and dev data
sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.train')
sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testa')
sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(data_folder + '/eng.testb')

# return corpus
return TaggedCorpus(sentences_train, sentences_dev, sentences_test)
```

The `TaggedCorpus` contains a bunch of useful helper functions. For instance, you can downsample the data by calling
`downsample()` and passing a ratio. So, if you normally get a corpus like this:

```python
original_corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
```

then you can downsample the corpus, simply like this: 

```python
downsampled_corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)
```

If you print both corpora, you see that the second one has been downsampled to 10% of the data. 

```python
print("--- 1 Original ---")
print(original_corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```

This should print: 

```console
--- 1 Original ---
TaggedCorpus: 14987 train + 3466 dev + 3684 test sentences

--- 2 Downsampled ---
TaggedCorpus: 1499 train + 347 dev + 369 test sentences
```


## Training a Model

Here is example code for a small NER model trained over CoNLL-03 data, using simple GloVe embeddings.
In this example, we downsample the data to 10% of the original data. 

```python
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import WordEmbeddings
import torch

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)  # remove the last bit to not downsample
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings. In this case, simple GloVe embeddings
embeddings = WordEmbeddings('glove')

# initialize sequence tagger
from flair.tagging_model import SequenceTaggerLSTM

tagger = SequenceTaggerLSTM(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                                use_crf=True)
                                                
# put model on cuda if GPU is available (i.e. much faster training)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain
trainer = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=False)

# run training for 5 epochs
trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=5, save_model=True,
              train_with_dev=True, anneal_mode=True)
```

Alternatively, try using a stacked embedding with charLM and glove, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments, 
check [here](/resources/docs/EXPERIMENTS.md). 