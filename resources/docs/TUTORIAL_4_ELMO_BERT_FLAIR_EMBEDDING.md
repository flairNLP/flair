# Tutorial 3: Word Embeddings

We provide a set of classes with which you can embed the words in sentences in various ways. This tutorial explains
how that works. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this 
library.  

## Flair Embeddings

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
| 'multi-forward'    | English, German, French, Italian, Dutch, Polish |  |
| 'multi-backward'    | English, German, French, Italian, Dutch, Polish |  |
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
| 'slovenian-backward' | Slovenian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Backward LM embeddings over various sources (Europarl, Wikipedia and OpenSubtitles2018) |
| 'bulgarian-forward'  | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Forward LM embeddings over various sources (Europarl, Wikipedia or SETimes) |
| 'bulgarian-backward' | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Backward LM embeddings over various sources (Europarl, Wikipedia or SETimes) |
| 'dutch-forward'    | Dutch | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Forward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'dutch-backward'    | Dutch | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Backward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'swedish-forward'    | Swedish | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Forward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'swedish-backward'    | Swedish | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Backward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'french-forward'    | French | Added by [@mhham](https://github.com/mhham): Forward LM embeddings over French Wikipedia |
| 'french-backward'    | French | Added by [@mhham](https://github.com/mhham): Backward LM embeddings over French Wikipedia |
| 'czech-forward'    | Czech | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Forward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'czech-backward'    | Czech | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Backward LM embeddings over various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'portuguese-forward'    | Portuguese | Added by [@ericlief](https://github.com/ericlief/language_models): Forward LM embeddings |
| 'portuguese-backward'    | Portuguese | Added by [@ericlief](https://github.com/ericlief/language_models): Backward LM embeddings |

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

## Next 

You can now either look into [document embeddings](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) to embed entire text 
passages with one vector for tasks such as text classification, or go directly to the tutorial about 
[loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which is a pre-requirement for
[training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).

