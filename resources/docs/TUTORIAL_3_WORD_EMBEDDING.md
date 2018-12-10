# Tutorial 3: Word Embeddings

We provide a set of classes with which you can embed the words in sentences in various ways. This tutorial explains
how that works. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this 
library.  


# Embeddings

All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which you need to 
call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains 
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are Pytorch vectors, so they can be immediately used for training and
fine-tuning.

## Classic Word Embeddings

Classic word embeddings are static and word-level, meaning that each distinct word gets exactly one pre-computed 
embedding. Most embeddings fall under this class, including the popular GloVe or Komnios embeddings. 

Simply instantiate the WordEmbeddings class and pass a string identifier of the embedding you wish to load. So, if 
you want to use GloVe embeddings, pass the string 'glove' to the constructor: 

```python
from flair.embeddings import WordEmbeddings

# init embedding
glove_embedding = WordEmbeddings('glove')
```
Now, create an example sentence and call the embedding's `embed()` method. You can also pass a list of sentences to
this method since some embedding types make use of batching to increase speed.

```python
# create sentence.
sentence = Sentence('The grass is green .')

# embed a sentence using glove.
glove_embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

This prints out the tokens and their embeddings. GloVe embeddings are Pytorch vectors of dimensionality 100.

You choose which pre-trained embeddings you load by passing the appropriate 
id string to the constructor of the `WordEmbeddings` class. Typically, you use
the **two-letter language code** to init an embedding, so 'en' for English and
'de' for German and so on. For English, we provide a few more options, so
here you can choose between instantiating 'en-glove', 'en-extvec' and so on.

The following embeddings are currently supported:
 
| ID | Language | Embedding | 
| ------------- | -------------  | ------------- |
| 'en-glove' (or 'glove') | English | GloVe embeddings |
| 'en-extvec' (or 'extvec') | English |Komnios embeddings |
| 'en-crawl' (or 'crawl')  | English | FastText embeddings over Web crawls |
| 'en' (or 'en-news' or 'news')  |English | FastText embeddings over news and wikipedia data |
| 'de' | German |German FastText embeddings |
| 'fr' | French | French FastText embeddings |
| 'pl' | Polish | Polish FastText embeddings |
| 'it' | Italian | Italian FastText embeddings |
| 'es' | Spanish | Spanish FastText embeddings |
| 'pt' | Portuguese | Portuguese FastText embeddings |
| 'nl' | Dutch | Dutch FastText embeddings |
| 'ar' | Arabic | Arabic FastText embeddings |
| 'sv' | Swedish | Swedish FastText embeddings |

So, if you want to load German FastText embeddings, instantiate as follows:

```python
german_embedding = WordEmbeddings('de')
```

We generally recommend the FastText embeddings, or GloVe if you want a smaller model.

If you want to use any other embeddings (not listed in the list above), you can load those by calling
```python
custom_embedding = WordEmbeddings('path/to/your/custom/embeddings.gensim')
```
If you want to load custom embeddings you need to make sure, that the custom embeddings are correctly formatted to
[gensim](https://radimrehurek.com/gensim/models/word2vec.html).

You can, for example, convert [FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) to gensim using the
following code snippet:
```python
import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('/path/to/fasttext/embeddings.txt', binary=False)
word_vectors.save('/path/to/converted')
```

## Contextual String Embeddings

Contextual string embeddings are [powerful embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and
thus fundamentally model words as sequences of characters. And (2) they are **contextualized** by their
surrounding text, meaning that the *same word will have different embeddings depending on its
contextual use*.

With Flair, you can use these embeddings simply by instantiating the appropriate embedding class, same as before:

```python
from flair.embeddings import CharLMEmbeddings

# init embedding
charlm_embedding_forward = CharLMEmbeddings('news-forward')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
charlm_embedding_forward.embed(sentence)
```

You choose which embeddings you load by passing the appropriate string to the constructor of the `CharLMEmbeddings` class. 
Currently, the following contextual string embeddings are provided (more coming):
 
| ID | Language | Embedding | 
| -------------     | ------------- | ------------- |
| 'news-forward'    | English | Forward LM embeddings over 1 billion word corpus |
| 'news-backward'   | English | Backward LM embeddings over 1 billion word corpus |
| 'news-forward-fast'    | English | Smaller, CPU-friendly forward LM embeddings over 1 billion word corpus |
| 'news-backward-fast'   | English | Smaller, CPU-friendly backward LM embeddings over 1 billion word corpus |
| 'mix-forward'     | English | Forward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'mix-backward'    | English | Backward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'german-forward'  | German  | Forward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'german-backward' | German  | Backward LM embeddings over mixed corpus (Web, Wikipedia, Subtitles) |
| 'polish-forward'  | Polish  | Added by [@borchmann](https://github.com/applicaai/poleval-2018): Forward LM embeddings over web crawls (Polish part of CommonCrawl) |
| 'polish-backward' | Polish  | Added by [@borchmann](https://github.com/applicaai/poleval-2018): Backward LM embeddings over web crawls (Polish part of CommonCrawl) |
| 'slovenian-forward'  | Slovenian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Forward LM embeddings over various sources (Europarl, Wikipedia and OpenSubtitles2018) |
| 'slovenian-backward' | Slovenian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms):Backward LM embeddings over various sources (Europarl, Wikipedia and OpenSubtitles2018) |
| 'bulgarian-forward'  | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms):Forward LM embeddings over various sources (Europarl, Wikipedia or SETimes) |
| 'bulgarian-backward' | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms):Backward LM embeddings over various sources (Europarl, Wikipedia or SETimes) |


So, if you want to load embeddings from the English news backward LM model, instantiate the method as follows:

```python
charlm_embedding_backward = CharLMEmbeddings('news-backward')
```

## BERT Embeddings

[BERT embeddings](https://arxiv.org/pdf/1810.04805.pdf) were developed by Devlin et al. (2018) and are a different kind
of powerful word embedding based on a bidirectional transformer architecture.
We are using the implementation of [huggingface](https://github.com/huggingface/pytorch-pretrained-BERT) in Flair.
The embeddings itself are wrapped into our simple embedding interface, so that they can be used like any other
embedding.

```python
from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

You can load any of the pre-trained BERT models by providing the model string during initialization:

| ID | Language | Embedding |
| -------------     | ------------- | ------------- |
| 'bert-base-uncased' | English | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-large-uncased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-cased'    | English | 12-layer, 768-hidden, 12-heads , 110M parameters |
| 'bert-large-cased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-multilingual-cased'     | 102 languages | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-base-chinese'    | Chinese Simplified and Traditional | 12-layer, 768-hidden, 12-heads, 110M parameters |


## ELMo Embeddings

[ELMo embeddings](http://www.aclweb.org/anthology/N18-1202) were presented by Peters et al. in 2018. They are using
a bidirectional recurrent neural network to predict the next word in a text.
We are using the implementation of [AllenNLP](https://allennlp.org/elmo). As this implementation comes with a lot of
sub-dependencies, you need to first install the library via `pip install allennlp` before you can use it in Flair.
Using the embeddings is as simple as using any other embedding type:

```python
from flair.embeddings import ELMoEmbeddings

# init embedding
embedding = ELMoEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

AllenNLP provides the following pre-trained models, that can be used. To use any of the following models inside Flair
simple specify the embedding id when initializing the `ELMoEmbeddings`.

| ID | Language | Embedding |
| ------------- | ------------- | ------------- |
| 'small' | English | 1024-hidden, 1 layer, 14.6M parameters |
| 'medium'   | English | 2048-hidden, 1 layer, 28.0M parameters |
| 'original'    | English | 4096-hidden, 2 layers, 93.6M parameters |
| 'pt'   | Portuguese | |


## Character Embeddings

Some embeddings - such as character-features - are not pre-trained but rather trained on the downstream task. Normally
this requires you to implement a [hierarchical embedding architecture](http://neuroner.com/NeuroNERengine_with_caption_no_figure.png). 

With Flair, you need not worry about such things. Just choose the appropriate
embedding class and character features will then automatically train during downstream task training. 

```python
from flair.embeddings import CharacterEmbeddings

# init embedding
embedding = CharacterEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

# Stacked Embeddings

Stacked embeddings are one of the most important concepts of this library. You can use them to combine different embeddings
together, for instance if you want to use both traditional embeddings together with contextual sting embeddings. 
Stacked embeddings allow you to mix and match. We find that a combination of embeddings often gives best results. 

All you need to do is use the `StackedEmbeddings` class and instantiate it by passing a list of embeddings that you wish 
to combine. For instance, lets combine classic GloVe embeddings with embeddings from a forward and backward 
character language model.

First, instantiate the three embeddings you wish to combine: 

```python
from flair.embeddings import WordEmbeddings, CharLMEmbeddings

# init GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init CharLM embeddings
charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these three embeddings.

```python
from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[glove_embedding, charlm_embedding_forward, charlm_embedding_backward])
```

That's it! Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
sentence = Sentence('The grass is green .')

# just embed a sentence using the StackedEmbedding as you would with any single embedding.
stacked_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

Words are now embedded using a concatenation of three different embeddings. This means that the resulting embedding
vector is still a single Pytorch vector. 

## Next 

You can now either look into [document embeddings](/resources/docs/TUTORIAL_4_DOCUMENT_EMBEDDINGS.md) to embed entire text 
passages with one vector for tasks such as text classification, or go directly to the tutorial about 
[training your own models](/resources/docs/TUTORIAL_5_TRAINING_A_MODEL.md). 

