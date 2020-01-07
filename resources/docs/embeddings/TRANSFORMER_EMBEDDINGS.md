# Transformers

Thanks to the brilliant [`transformers`](https://github.com/huggingface/transformers) library from [Hugging Face](https://github.com/huggingface),
Flair is able to support various Transformer-based architectures like BERT or XLNet.

The following embeddings can be used in Flair:

* `BertEmbeddings`
* `OpenAIGPTEmbeddings`
* `OpenAIGPT2Embeddings`
* `TransformerXLEmbeddings`
* `XLNetEmbeddings`
* `XLMEmbeddings`
* `RoBERTaEmbeddings`
* `CamembertEmbeddings`
* `XLMRobertaEmbeddings`

This section shows how to use these Transformer-based architectures in Flair.

## BERT Embeddings

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
| -------------------- | ------------------- | ------------------------------------------------------------------------------
| `bert_model_or_path` | `bert-base-uncased` | Defines BERT model or points to user-defined path, see [model section](#Models)
| `layers`             | `-1,-2,-3,-4`       | Defines the to be used layers of the Transformer-based model
| `pooling_operation`  | `first`             | See [Pooling operation section](#Pooling-operation).
| `use_scalar_mix`     | `False`             | See [Scalar mix section](#Scalar-mix).

You can load any of the pre-trained BERT models by providing `bert_model_or_path` during initialization.

### DistilBERT

It is also possible to use [distilled versions](https://medium.com/huggingface/distilbert-8cf3380435b5)
of BERT (DistilBERT). For available DistilBERT models, proposed by [Sanh et. al (2019)](https://arxiv.org/abs/1910.01108),
please have a look at the [model section](#Models).

The following example shows how to use a DistilBERT model:

```python
from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings(bert_model_or_path="distilbert-base-german-cased")

# create a sentence
sentence = Sentence('Es lohnt sich, Berlin und München zu besichtigen !')

# embed words in sentence
embedding.embed(sentence)
```

### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

The ALBERT model, proposed by [Lan et. al (2019)](https://arxiv.org/abs/1909.11942) can also be used:

```python
from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings(bert_model_or_path="albert-base-v2")

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

All available ALBERT models can be found in the [model section](#Models).

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
| ------------------------------- | ------------ | ------------------------------------------------------------
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
| ------------------------------- | ------------- | -----------------------------------------------------------------
| `pretrained_model_name_or_path` | `gpt2-medium` | Defines name or path of GPT-2 model, see [model section](#Models)
| `layers`                        | `1`           | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`  | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`       | See [Scalar mix section](#Scalar-mix)

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
| ------------------------------- | ------------------ | ------------------------------------------------------------
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
| ------------------------------- | ------------------- | -----------------------------------------------------------------
| `pretrained_model_name_or_path` | `xlnet-large-cased` | Defines name or path of XLNet model, see [model section](#Models)
| `layers`                        | `1`                 | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`        | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`             | See [Scalar mix section](#Scalar-mix)

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
| ------------------------------- | ------------------- | ---------------------------------------------------------------
| `pretrained_model_name_or_path` | `xlm-mlm-en-2048`   | Defines name or path of XLM model, see [model section](#Models)
| `layers`                        | `1`                 | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first_last`        | See [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`             | See [Scalar mix section](#Scalar-mix)

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
| ------------------------------- | --------------- | -------------------------------------------------------------------
| `pretrained_model_name_or_path` | `roberta-base`  | Defines name or path of RoBERTa model, see [model section](#Models)
| `layers`                        | `-1`            | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first`         | [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`         | [Scalar mix section](#Scalar-mix)

## CamemBERT Embeddings

CamemBERT (a Tasty French Language Model) model was proposed by [Martin et. al (2019)](https://arxiv.org/abs/1911.03894),
and was trained on a large French corpus.

It can be used with the `CamembertEmbeddings` class:

```python
from flair.embeddings import CamembertEmbeddings

# init embedding
embedding = CamembertEmbeddings()

# create a sentence
sentence = Sentence("J'aime le camembert !")

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `CamembertEmbeddings` class:

| Argument                        | Default          | Description
| ------------------------------- | ---------------- | ---------------------------------------------------------------------
| `pretrained_model_name_or_path` | `camembert-base` | Defines name or path of CamemBERT model, see [model section](#Models)
| `layers`                        | `-1`             | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first`          | [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`          | [Scalar mix section](#Scalar-mix)

## XLMRobertaEmbeddings

XLM-RoBERTa, proposed by [Conneau et. al (2019)](https://arxiv.org/abs/1911.02116) in the "Unsupervised Cross-lingual Representation Learning at Scale"
paper, is a multi-lingual language model trained on 2.5TB of filtered CommonCrawl data.

It can be used with the `XLMRobertaEmbeddings` class:

```python
from flair.embeddings import XLMRobertaEmbeddings

# init embedding
embedding = XLMRobertaEmbeddings()

# create a sentence
sentence = Sentence("J'aime le camembert, and best whishes from München und Berlin .")

# embed words in sentence
embedding.embed(sentence)
```

The following arguments can be passed to the `XLMRobertaEmbeddings` class:

| Argument                        | Default            | Description
| ------------------------------- | ------------------- | -----------------------------------------------------------------
| `pretrained_model_name_or_path` | `xlm-roberta-large` | Defines name or path of XLM-R model, see [model section](#Models)
| `layers`                        | `-1`                | Defines the to be used layers of the Transformer-based model
| `pooling_operation`             | `first`             | [Pooling operation section](#Pooling-operation)
| `use_scalar_mix`                | `False`             | [Scalar mix section](#Scalar-mix)

### Pooling operation

Most of the Transformer-based models (except Transformer-XL) use subword tokenization. E.g. the following
token `puppeteer` could be tokenized into the subwords: `pupp`, `##ete` and `##er`.

We implement different pooling operations for these subwords to generate the final token representation:

* `first`: only the embedding of the first subword is used
* `last`: only the embedding of the last subword is used
* `first_last`: embeddings of the first and last subwords are concatenated and used
* `mean`: a `torch.mean` over all subword embeddings is calculated and used

## Scalar mix

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

## Models

Please have a look at the awesome Hugging Face [documentation](https://huggingface.co/transformers/v2.3.0/pretrained_models.html)
for all supported pretrained models!
