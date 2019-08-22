# Tutorial 4: BERT, ELMo, and Flair Embeddings

Next to standard WordEmbeddings and CharacterEmbeddings, we also provide classes for BERT, ELMo and Flair embeddings. These embeddings enable you to train truly state-of-the-art NLP models.

This tutorial explains how to use these embeddings. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library as well as [standard word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md), in particular the `StackedEmbeddings` class.

## Embeddings

All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which you need to
call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are Pytorch vectors, so they can be immediately used for training and
fine-tuning.

## Flair Embeddings

Contextual string embeddings are [powerful embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and
thus fundamentally model words as sequences of characters. And (2) they are **contextualized** by their
surrounding text, meaning that the *same word will have different embeddings depending on its
contextual use*.

With Flair, you can use these embeddings simply by instantiating the appropriate embedding class, same as standard word embeddings:

```python
from flair.embeddings import FlairEmbeddings

# init embedding
flair_embedding_forward = FlairEmbeddings('news-forward')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
flair_embedding_forward.embed(sentence)
```

You choose which embeddings you load by passing the appropriate string to the constructor of the `FlairEmbeddings` class.
Currently, the following contextual string embeddings are provided (note: replace '*X*' with either '*forward*' or '*backward*'):

| ID | Language | Embedding |
| -------------     | ------------- | ------------- |
| 'multi-X'    | English, German, French, Italian, Dutch, Polish | Mix of corpora (Web, Wikipedia, Subtitles, News) |
| 'multi-X-fast'    | English, German, French, Italian, Dutch, Polish | Mix of corpora (Web, Wikipedia, Subtitles, News), CPU-friendly |
| 'news-X'    | English | Trained with 1 billion word corpus |
| 'news-X-fast'    | English | Trained with 1 billion word corpus, CPU-friendly |
| 'mix-X'     | English | Trained with mixed corpus (Web, Wikipedia, Subtitles) |
| 'ar-X'     | Arabic | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'bg-X'  | Bulgarian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'bg-X-fast'  | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or SETimes) |
| 'cs-X'     | Czech | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'cs-v0-X'    | Czech | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'de-X'  | German  | Trained with mixed corpus (Web, Wikipedia, Subtitles) |
| 'de-historic-ha-X'  | German (historical) | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Hamburger Anzeiger* |
| 'de-historic-wz-X'  | German (historical) | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Wiener Zeitung* |
| 'es-X'    | Spanish | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipedia |
| 'es-X-fast'    | Spanish | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipediam CPU-friendly |
| 'eu-X'    | Basque | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'eu-v0-X'    | Basque | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'fa-X'     | Persian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'fi-X'     | Finnish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'fr-X'    | French | Added by [@mhham](https://github.com/mhham): Trained with French Wikipedia |
| 'he-X'     | Hebrew | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'hi-X'     | Hindi | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'hr-X'     | Croatian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'id-X'     | Indonesian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'it-X'     | Italian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'ja-X'    | Japanese | Added by [@frtacoa](https://github.com/zalandoresearch/flair/issues/527): Trained with 439M words of Japanese Web crawls (2048 hidden states, 2 layers)|
| 'nl-X'     | Dutch | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'nl-v0-X'    | Dutch | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'no-X'     | Norwegian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'pl-X'  | Polish  | Added by [@borchmann](https://github.com/applicaai/poleval-2018): Trained with web crawls (Polish part of CommonCrawl) |
| 'pl-opus-X'     | Polish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'pt-X'    | Portuguese | Added by [@ericlief](https://github.com/ericlief/language_models): LM embeddings |
| 'sl-X'     | Slovenian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'sl-v0-X'  | Slovenian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia and OpenSubtitles2018) |
| 'sv-X'    | Swedish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'sv-v0-X'    | Swedish | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'pubmed-X'    | English | Added by [@jessepeng](https://github.com/zalandoresearch/flair/pull/519): Trained with 5% of PubMed abstracts until 2015 (1150 hidden states, 3 layers)|


So, if you want to load embeddings from the German forward LM model, instantiate the method as follows:

```python
flair_de_forward = FlairEmbeddings('de-forward')
```

And if you want to load embeddings from the Bulgarian backward LM model, instantiate the method as follows:

```python
flair_bg_backward = FlairEmbeddings('bg-backward')
```

## Recommended Flair Usage

We recommend combining both forward and backward Flair embeddings. Depending on the task, we also recommend adding standard word embeddings into the mix. So, our recommended `StackedEmbedding` for most English tasks is:


```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                       ])
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
Words are now embedded using a concatenation of three different embeddings. This combination often gives state-of-the-art accuracy.


## PyTorch-Transformers

Thanks to the brilliant [`pytorch-transformers`](https://github.com/huggingface/pytorch-transformers) library from [Hugging Face](https://github.com/huggingface),
Flair is able to support various Transformer-based architectures like BERT or XLNet.

The following embeddings can be used in Flair:

* `BertEmbeddings`
* `OpenAIGPTEmbeddings`
* `OpenAIGPT2Embeddings`
* `TransformerXLEmbeddings`
* `XLNetEmbeddings`
* `XLMEmbeddings`
* `RoBERTaEmbeddings`

This section shows how to use these Transformer-based architectures in Flair and is heavily based on the excellent
[PyTorch-Transformers pre-trained models documentation](https://huggingface.co/pytorch-transformers/pretrained_models.html).

### BERT Embeddings

[BERT embeddings](https://arxiv.org/pdf/1810.04805.pdf) were developed by Devlin et al. (2018) and are a different kind
of powerful word embedding based on a bidirectional transformer architecture.
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

The `BertEmbeddings` class has several arguments:

| Argument             | Default             | Description
| -------------------- | ------------------- | -------------------------------------------------
| `bert_model_or_path` | `bert-base-uncased` | Defines BERT model or points to user-defined path
| `layers`             | `-1,-2,-3,-4`       | Defines the to be used layers of the Transformer-based model
| `pooling_operation`  | `first`             | See [Pooling operation section](#Pooling-operation).
| `use_scalar_mix`     | `False`             | See [Scalar mix section](#Scalar-mix).

You can load any of the pre-trained BERT models by providing `bert_model_or_path` during initialization:

| Model                                                   | Details
| ------------------------------------------------------- | -----------------------------------------------
| `bert-base-uncased`                                     | 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on lower-cased English text
| `bert-large-uncased`                                    | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | Trained on lower-cased English text
| `bert-base-cased`                                       | 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on cased English text
| `bert-large-cased`                                      | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | Trained on cased English text
| `bert-base-multilingual-uncased`                        | (Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on lower-cased text in the top 102 languages with the largest Wikipedias
|                                                         | (see [details](https://github.com/google-research/bert/blob/master/multilingual.md))
| `bert-base-multilingual-cased`                          | (New, **recommended**) 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on cased text in the top 104 languages with the largest Wikipedias
|                                                         | (see [details](https://github.com/google-research/bert/blob/master/multilingual.md))
| `bert-base-chinese`                                     | 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on cased Chinese Simplified and Traditional text
| `bert-base-german-cased`                                | 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | Trained on cased German text by Deepset.ai
|                                                         | (see [details on deepset.ai website](https://deepset.ai/german-bert))
| `bert-large-uncased-whole-word-masking`                 | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | Trained on lower-cased English text using Whole-Word-Masking
|                                                         | (see [details](https://github.com/google-research/bert/#bert))
| `bert-large-cased-whole-word-masking`                   | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | Trained on cased English text using Whole-Word-Masking
|                                                         | (see [details](https://github.com/google-research/bert/#bert))
| `bert-large-uncased-whole-word-masking-finetuned-squad` | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | The `bert-large-uncased-whole-word-masking` model fine-tuned on SQuAD (see details of fine-tuning in the
|                                                         | [example section of PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers/tree/master/examples))
| `bert-large-cased-whole-word-masking-finetuned-squad`   | 24-layer, 1024-hidden, 16-heads, 340M parameters
|                                                         | The `bert-large-cased-whole-word-masking` model fine-tuned on SQuAD
|                                                         | (see [details of fine-tuning in the example section](https://huggingface.co/pytorch-transformers/examples.html))
| `bert-base-cased-finetuned-mrpc`                        | 12-layer, 768-hidden, 12-heads, 110M parameters
|                                                         | The `bert-base-cased` model fine-tuned on MRPC
|                                                         | (see [details of fine-tuning in the example section of PyTorch-Transformers](https://huggingface.co/pytorch-transformers/examples.html))

## OpenAI GPT Embeddings

The OpenAI GPT model was proposed by [Radford et. al (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).
GPT is an uni-directional Transformer-based model.

The following example shows how to use the `OpenAIGPTEmbeddings`:

```python
from flair.embeddings import OpenAIGPTEmbeddings

# init embedding
embedding = OpenAIGPTEmbeddings()

# create a sentence
sentence = Sentence('Berlin and Munich are nice cities .')

# embed words in sentence
embedding.embed(sentence)
```

The `OpenAIGPTEmbeddings` class has several arguments:

| Argument                        | Default      | Description
| ------------------------------- | ------------ | -------------------------------------------------
| `pretrained_model_name_or_path` | `openai-gpt` | Defines name or path of GPT model
| `layers`                        | `1`          | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last` | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`      | See [Scalar mix section](#Scalar-mix)

## OpenAI GPT-2 Embeddings

The OpenAI GPT-2 model was proposed by [Radford et. al (2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
GPT-2 is also an uni-directional Transformer-based model, that was trained on a larger corpus compared to the GPT model.

The GPT-2 model can be used with the `OpenAIGPT2Embeddings` class:

```python
from flair.embeddings import OpenAIGPT2Embeddings

# init embedding
embedding = OpenAIGPT2Embeddings()

# create a sentence
sentence = Sentence('The Englischer Garten is a large public park in the centre of Munich .')

# embed words in sentence
embedding.embed(sentence)
```

The `OpenAIGPT2Embeddings` class has several arguments:

| Argument                        | Default       | Description
| ------------------------------- | ------------- | -------------------------------------------------
| `pretrained_model_name_or_path` | `gpt2-medium` | Defines name or path of GPT-2 model
| `layers`                        | `1`           | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`  | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`       | See [Scalar mix section](#Scalar-mix)

Following GPT-2 models can be used:

| Model         | Details
| ------------- | -----------------------------------------------
| `gpt2`        | 12-layer, 768-hidden, 12-heads, 117M parameters
|               | OpenAI GPT-2 English model
| `gpt2-medium` | 24-layer, 1024-hidden, 16-heads, 345M parameters
|               | OpenAI's Medium-sized GPT-2 English model

## Transformer-XL Embeddings

The Transformer-XL model was proposed by [Dai et. al (2019)](https://arxiv.org/abs/1901.02860).
It is an uni-directional Transformer-based model with relative positioning embeddings.

The Transformer-XL model can be used with the `TransformerXLEmbeddings` class:

```python
from flair.embeddings import TransformerXLEmbeddings

# init embedding
embedding = TransformerXLEmbeddings()

# create a sentence
sentence = Sentence('The Berlin Zoological Garden is the oldest and best-known zoo in Germany .')

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `TransformerXLEmbeddings` class:

| Argument                        | Default            | Description
| ------------------------------- | ------------------ | -------------------------------------------------
| `pretrained_model_name_or_path` | `transfo-xl-wt103` | Defines name or path of Transformer-XL model
| `layers`                        | `1,2,3`            | Defines the to be used layers of the Transformer-based model
| `use_scalar_mix`                | `False`            | See [Scalar mix section](#Scalar-mix)

Notice: The Transformer-XL model (trained on WikiText-103) is a word-based language model. Thus, no subword tokenization
is necessary is needed (`pooling_operation` is not needed).

## XLNet Embeddings

The XLNet model was proposed by [Yang et. al (2019)](https://arxiv.org/abs/1906.08237).
It is an extension of the Transformer-XL model using an autoregressive method to learn bi-directional contexts.

The XLNet model can be used with the `XLNetEmbeddings` class:

```python
from flair.embeddings import XLNetEmbeddings

# init embedding
embedding = XLNetEmbeddings()

# create a sentence
sentence = Sentence('The Hofbräuhaus is a beer hall in Munich .')

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `XLNetEmbeddings` class:

| Argument                        | Default             | Description
| ------------------------------- | ------------------- | -------------------------------------------------
| `pretrained_model_name_or_path` | `xlnet-large-cased` | Defines name or path of XLNet model
| `layers`                        | `1`                 | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`        | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`             | See [Scalar mix section](#Scalar-mix)

Following XLNet models can be used:

| Model              | Details
| ------------------ | -----------------------------------------------
| `xlnet-base-cased` | 12-layer, 768-hidden, 12-heads, 110M parameters
|                    | XLNet English model
| `xlnet-large-cased`| 24-layer, 1024-hidden, 16-heads, 340M parameters
|                    | XLNet Large English model

## XLM Embeddings

The XLM model was proposed by [Lample and Conneau (2019)](https://arxiv.org/abs/1901.07291).
It extends the generative pre-training approach for English to multiple languages and show the effectiveness of
cross-lingual pretraining.

The XLM model can be used with the `XLMEmbeddings` class:

```python
from flair.embeddings import XLMEmbeddings

# init embedding
embedding = XLMEmbeddings()

# create a sentence
sentence = Sentence('The BER is an international airport under construction near Berlin .')

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `XLMEmbeddings` class:

| Argument                        | Default             | Description
| ------------------------------- | ------------------- | -------------------------------------------------
| `pretrained_model_name_or_path` | `xlm-mlm-en-2048`   | Defines name or path of XLM model
| `layers`                        | `1`                 | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`        | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`             | See [Scalar mix section](#Scalar-mix)

Following XLM models can be used:

| Model                     | Details
| ------------------------- | -------------------------------------------------------------------------------------------------------
| `xlm-mlm-en-2048`         | 12-layer, 1024-hidden, 8-heads
|                           | XLM English model
| `xlm-mlm-ende-1024`       | 6-layer, 1024-hidden, 8-heads
|                           | XLM English-German Multi-language model
| `xlm-mlm-enfr-1024`       | 6-layer, 1024-hidden, 8-heads
|                           | XLM English-French Multi-language model
| `xlm-mlm-enro-1024`       | 6-layer, 1024-hidden, 8-heads
|                           | XLM English-Romanian Multi-language model
| `xlm-mlm-xnli15-1024`     | 12-layer, 1024-hidden, 8-heads
|                           | XLM Model pre-trained with MLM on the [15 XNLI languages](https://github.com/facebookresearch/XNLI)
| `xlm-mlm-tlm-xnli15-1024` | 12-layer, 1024-hidden, 8-heads
|                           | XLM Model pre-trained with MLM + TLM on the [15 XNLI languages](https://github.com/facebookresearch/XNLI)
| `xlm-clm-enfr-1024`       | 12-layer, 1024-hidden, 8-heads
|                           | XLM English model trained with CLM (Causal Language Modeling)
| `xlm-clm-ende-1024`       | 6-layer, 1024-hidden, 8-heads
|                           | XLM English-German Multi-language model trained with CLM (Causal Language Modeling)

## RoBERTa Embeddings

The RoBERTa (**R**obustly **o**ptimized **BERT** pre-training **a**pproach) model was proposed by [Liu et. al (2019)](https://arxiv.org/abs/1907.11692),
and uses an improved pre-training procedure to train a BERT model on a large corpus.

It can be used with the `RoBERTaEmbeddings` class:

```python
from flair.embeddings import RoBERTaEmbeddings

# init embedding
embedding = RoBERTaEmbeddings()

# create a sentence
sentence = Sentence("The Oktoberfest is the world's largest Volksfest .")

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `RoBERTaEmbeddings` class:

| Argument                        | Default         | Description
| ------------------------------- | --------------- | -------------------------------------------------
| `pretrained_model_name_or_path` | `roberta-base`  | Defines name or path of RoBERTa model
| `layers`                        | `-1`            | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first`         | [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`         | [Scalar mix section](#Scalar-mix)

Following XLM models can be used:

| Model                | Details
| -------------------- | -------------------------------------------------------------------------------------------------------
| `roberta-base`       | 12-layer, 768-hidden, 12-heads
|                      | RoBERTa English model
| `roberta-large`      | 24-layer, 1024-hidden, 16-heads
|                      | RoBERTa English model
| `roberta-large-mnli` | 24-layer, 1024-hidden, 16-heads
|                      | RoBERTa English model, finetuned on MNLI

### Pooling operation

Most of the Transformer-based models (except Transformer-XL) use subword tokenization. E.g. the following
token `puppeteer` could be tokenized into the subwords: `pupp`, `##ete` and `##er`.

We implement different pooling operations for these subwords to generate the final token representation:

* `first`: only the embedding of the first subword is used
* `last`: only the embedding of the last subword is used
* `first_last`: embeddings of the first and last subwords are concatenated and used
* `mean`: a `torch.mean` over all subword embeddings is calculated and used

### Scalar mix

The Transformer-based models have a certain number of layers. [Liu et. al (2019)](https://arxiv.org/abs/1903.08855)
propose a technique called scalar mix, that computes a parameterised scalar mixture of user-defined layers.

This technique is very useful, because for some downstream tasks like NER or PoS tagging it can be unclear which
layer(s) of a Transformer-based model perform well, and per-layer analysis can take a lot of time.

To use scalar mix, all Transformer-based embeddings in Flair come with a `use_scalar_mix` argument. The following
example shows how to use scalar mix for a base RoBERTa model on all layers:

```python
from flair.embeddings import RoBERTaEmbeddings

# init embedding
embedding = RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-base", layers="0,1,2,3,4,5,6,7,8,9,10,11,12",
                              pooling_operation="first", use_scalar_mix=True)

# create a sentence
sentence = Sentence("The Oktoberfest is the world's largest Volksfest .")

# embed words in sentence
embedding.embed(sentence)
```

## ELMo Embeddings

[ELMo embeddings](http://www.aclweb.org/anthology/N18-1202) were presented by Peters et al. in 2018. They are using
a bidirectional recurrent neural network to predict the next word in a text.
We are using the implementation of [AllenNLP](https://allennlp.org/elmo). As this implementation comes with a lot of
sub-dependencies, which we don't want to include in Flair, you need to first install the library via
`pip install allennlp` before you can use it in Flair.
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

AllenNLP provides the following pre-trained models. To use any of the following models inside Flair
simple specify the embedding id when initializing the `ELMoEmbeddings`.

| ID | Language | Embedding |
| ------------- | ------------- | ------------- |
| 'small' | English | 1024-hidden, 1 layer, 14.6M parameters |
| 'medium'   | English | 2048-hidden, 1 layer, 28.0M parameters |
| 'original'    | English | 4096-hidden, 2 layers, 93.6M parameters |
| 'pt'   | Portuguese | |
| 'pubmed' | English biomedical data | [more information](https://allennlp.org/elmo) |


## Combining BERT and Flair

You can very easily mix and match Flair, ELMo, BERT and classic word embeddings. All you need to do is instantiate each embedding you wish to combine and use them in a StackedEmbedding.

For instance, let's say we want to combine the multilingual Flair and BERT embeddings to train a hyper-powerful multilingual downstream task model.

First, instantiate the embeddings you wish to combine:

```python
from flair.embeddings import FlairEmbeddings, BertEmbeddings

# init Flair embeddings
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these three embeddings.

```python
from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])
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
vector is still a single PyTorch vector.


## Next

You can now either look into [document embeddings](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) to embed entire text
passages with one vector for tasks such as text classification, or go directly to the tutorial about
[loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which is a pre-requirement for
[training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).

