# Tutorial 4: List of All Word Embeddings

This is not so much a tutorial, but rather a list of all embeddings that we currently support in Flair. Click on each embedding in the table below to get usage instructions. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library as well as [standard word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md), in particular the `StackedEmbeddings` class.

## Overview 

All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which you need to
call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

The following word embeddings are currently supported: 

| Class | Type | Paper | 
| ------------- | -------------  | -------------  | 
| [`BertEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from pretrained BERT | [Devlin et al. (2018)](https://www.aclweb.org/anthology/N19-1423/) |  
| [`BytePairEmbeddings`](/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md) | Subword-level word embeddings | [Heinzerling and Strube (2018)](https://www.aclweb.org/anthology/L18-1473)  |
| [`CharacterEmbeddings`](/resources/docs/embeddings/CHARACTER_EMBEDDINGS.md) | Task-trained character-level embeddings of words | [Lample et al. (2016)](https://www.aclweb.org/anthology/N16-1030) |
| [`ELMoEmbeddings`](/resources/docs/embeddings/ELMO_EMBEDDINGS.md) | Contextualized word-level embeddings | [Peters et al. (2018)](https://aclweb.org/anthology/N18-1202)  |
| [`FastTextEmbeddings`](/resources/docs/embeddings/FASTTEXT_EMBEDDINGS.md) | Word embeddings with subword features | [Bojanowski et al. (2017)](https://aclweb.org/anthology/Q17-1010)  |
| [`FlairEmbeddings`](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md) | Contextualized character-level embeddings | [Akbik et al. (2018)](https://www.aclweb.org/anthology/C18-1139/)  |
| [`PooledFlairEmbeddings`](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md) | Pooled variant of `FlairEmbeddings` |  [Akbik et al. (2019)](https://www.aclweb.org/anthology/N19-1078/)  |
| [`OpenAIGPTEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) and [`OpenAIGPT2Embeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from pretrained OpenAIGPT models | [Radford et al. (2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |  
| [`RoBERTaEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from RoBERTa | [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) |  
| [`TransformerXLEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from pretrained transformer-XL | [Dai et al. (2019)](https://arxiv.org/abs/1901.02860) |  
| [`WordEmbeddings`](/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md) | Classic word embeddings |  |
| [`XLNetEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from pretrained XLNet | [Yang et al. (2019)](https://arxiv.org/abs/1906.08237) |  
| [`XLMEmbeddings`](/resources/docs/embeddings/TRANSFOMER_EMBEDDINGS.md) | Embeddings from pretrained XLM | [Lample and Conneau (2019)](https://arxiv.org/abs/1901.07291) |  


## Combining BERT and Flair

You can very easily mix and match Flair, ELMo, BERT and classic word embeddings. All you need to do is instantiate each embedding you wish to combine and use them in a `StackedEmbedding`.

For instance, let's say we want to combine the multilingual Flair and BERT embeddings to train a hyper-powerful multilingual downstream task model. First, instantiate the embeddings you wish to combine:

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

