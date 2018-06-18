# Tutorial

Let's look into some core functionality to understand the library better.

## NLP base types

First, you need to construct Sentence objects for your text.

```python
# The sentence objects holds a sentence that we may want to embed
from flair.data import Sentence

# Make a sentence object by passing a whitespace tokenized string
sentence = Sentence('The grass is green .')

# Print the object to see what's in there
print(sentence)

# The Sentence object has a list of Token objects (each token represents a word)
for token in sentence.tokens:
    print(token)

# add a tag to a word in the sentence
sentence.get_token(4).add_tag('ner', 'color')

# print the sentence with all tags of this type
print(sentence.to_tag_string('ner'))

```

## Embeddings

### Classic Word Embeddings

Now, you can embed the words in a sentence. We start with a simple example that uses GloVe embeddings:

```python

# all embeddings inherit from the TextEmbeddings class. Init a simple glove embedding.
from flair.embeddings import WordEmbeddings
glove_embedding = WordEmbeddings('glove')

# embed a sentence using glove.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
glove_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

### Contextual String Embeddings

You can also use our contextual string embeddings. Same code as above, with different TextEmbedding class:

```python

# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import CharLMEmbeddings
contextual_string_embedding = CharLMEmbeddings('news-forward')

# embed a sentence using CharLM.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
contextual_string_embedding.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```


### Stacked Embeddings

Very often, you want to combine different embedding types. For instance, you might want to combine classic static
word embeddings such as GloVe with embeddings from a forward and backward language model. This normally gives best
results.

For this case, use the StackedEmbeddings class which combines a list of TextEmbeddings.

```python

# the CharLMEmbedding also inherits from the TextEmbeddings class
from flair.embeddings import WordEmbeddings, CharLMEmbeddings

# init GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init CharLM embedding
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')

# now create the StackedEmbedding object that combines all embeddings
from flair.embeddings import StackedEmbeddings
stacked_embeddings = StackedEmbeddings(embeddings=[glove_embedding, charlm_embedding_forward, charlm_embedding_backward])


# just embed a sentence using the StackedEmbedding as you would with any single embedding.
from flair.data import Sentence
sentence = Sentence('The grass is green .')
stacked_embeddings.get_embeddings(sentences=[sentence])

# now check out the embedded tokens.
for token in sentence.tokens:
    print(token)
    print(token.get_embedding())
```

## Training a Model

Here is example code for a small NER model trained over CoNLL-03 data, using simple GloVe embeddings.
In this example, we downsample the data to 10% of the original data. 

```python
from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import WordEmbeddings
import torch

# 1. get the corpus
task_data_fetcher: NLPTaskDataFetcher = NLPTaskDataFetcher()
corpus: TaggedCorpus = task_data_fetcher.fetch_data(NLPTask.CONLL_03).downsample(0.1)  # remove the last bit to not downsample
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
                                                use_crf=False)
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer = TagTrain(tagger, corpus, tag_type=tag_type, test_mode=True)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=5, save_model=True,
              train_with_dev=True, anneal_mode=True)
```

Alternatively, try using a stacked embedding with charLM and glove, over the full data, for 150 epochs.
This will give you the state-of-the-art accuracy we report in the paper. To see the full code to reproduce experiments, 
check [here](/resources/docs/EXPERIMENTS.md). 