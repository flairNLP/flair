# Tutorial 3: Embeddings

This tutorial shows you how to use Flair to produce **embeddings** for words and documents. Embeddings
are vector representations that are useful for a variety of reasons. All Flair models are trained on 
top of embeddings, so if you want to train your own models, you should understand how embeddings work.

## Example 1: Embeddings Words with Transformers

Let's use a standard BERT model (bert-base-uncased) to embed the sentence "the grass is green".

Simply instantate `TransformerWordEmbeddings` and call `embed()` over an example sentence: 

```python
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

# init embedding
embedding = TransformerWordEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

This will cause **each word in the sentence** to be embedded. You can iterate through the words and get 
each embedding like this:

```python
# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

This will print each token as a long PyTorch vector: 
```console
Token[0]: "The"
tensor([-0.0323, -0.3904, -1.1946,  0.1296,  0.5806, ..], device='cuda:0')
Token[1]: "grass"
tensor([-0.3973,  0.2652, -0.1337,  0.4473,  1.1641, ..], device='cuda:0')
Token[2]: "is"
tensor([ 0.1374, -0.3688, -0.8292, -0.4068,  0.7717, ..], device='cuda:0')
Token[3]: "green"
tensor([-0.7722, -0.1152,  0.3661,  0.3570,  0.6573, ..], device='cuda:0')
Token[4]: "."
tensor([ 0.1441, -0.1772, -0.5911,  0.2236, -0.0497, ..], device='cuda:0')
```

*(Output truncated for readability, actually the vectors are much longer.)*

Transformer word embeddings are the most important concept in Flair. Check out more info in this dedicated chapter.

## Example 2: Embeddings Documents with Transformers

Sometimes you want to have an **embedding for a whole document**, not only individual words. In this case, use one of the 
DocumentEmbeddings classes in Flair. 

Let's again use a standard BERT model to get an embedding for the entire sentence "the grass is green":  

```python
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

# init embedding
embedding = TransformerDocumentEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

Now, the whole sentence is embedded. Print the embedding like this: 

```python
# now check out the embedded sentence
print(sentence.embedding)
```

Transformer document embeddings are the most important concept in Flair. Check out more info in this dedicated chapter.


## How to Stack Embeddings

Stacked embeddings are one of the most important concepts of this library. You can use them to combine different
embeddings together, for instance if you want to use both traditional embeddings together with contextual string
embeddings. Stacked embeddings allow you to mix and match. We find that a combination of embeddings gives best results, when not fine-tuning.

All you need to do is use the `StackedEmbeddings` class and instantiate it by passing a list of embeddings that you wish
to combine. For instance, lets combine classic GloVe embeddings with forward and backward Flair embeddings. This is a combination that we generally recommend to most users, especially for sequence labeling.

First, instantiate the two embeddings you wish to combine:

```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# init standard GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init Flair forward and backwards embeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these two embeddings.

```python
from flair.embeddings import StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
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

Words are now embedded using a concatenation of three different embeddings. This means that the resulting embedding
vector is still a single PyTorch vector.


## List of All Embeddings

The following embeddings are currently supported. The most important embeddings for this library are highlighted
bold:

| Class                                                                                       | Type                                                                                                                                                    | Paper |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------| -------------  |
| [`BytePairEmbeddings`](/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md)                  | Subword-level word embeddings                                                                                                                           | [Heinzerling and Strube (2018)](https://www.aclweb.org/anthology/L18-1473)  |
| [`CharacterEmbeddings`](/resources/docs/embeddings/CHARACTER_EMBEDDINGS.md)                 | Task-trained character-level embeddings of words                                                                                                        | [Lample et al. (2016)](https://www.aclweb.org/anthology/N16-1030) |
| [`ELMoEmbeddings`](/resources/docs/embeddings/ELMO_EMBEDDINGS.md)                           | Contextualized word-level embeddings                                                                                                                    | [Peters et al. (2018)](https://aclweb.org/anthology/N18-1202)  |
| [`FastTextEmbeddings`](/resources/docs/embeddings/FASTTEXT_EMBEDDINGS.md)                   | Word embeddings with subword features                                                                                                                   | [Bojanowski et al. (2017)](https://aclweb.org/anthology/Q17-1010)  |
| [**`FlairEmbeddings`**](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)                     | Contextualized character-level embeddings                                                                                                               | [Akbik et al. (2018)](https://www.aclweb.org/anthology/C18-1139/)  |
| [`OneHotEmbeddings`](/resources/docs/embeddings/ONE_HOT_EMBEDDINGS.md)                      | Standard one-hot embeddings of text or tags                                                                                                             | - |
| [`PooledFlairEmbeddings`](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)                   | Pooled variant of `FlairEmbeddings`                                                                                                                     |  [Akbik et al. (2019)](https://www.aclweb.org/anthology/N19-1078/)  |
| [**`TransformerWordEmbeddings`**](/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md)     | Embeddings from pretrained [transformers](https://huggingface.co/transformers/pretrained_models.html) (BERT, XLM, GPT, RoBERTa, XLNet, DistilBERT etc.) | [Devlin et al. (2018)](https://www.aclweb.org/anthology/N19-1423/) [Radford et al. (2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) [Dai et al. (2019)](https://arxiv.org/abs/1901.02860) [Yang et al. (2019)](https://arxiv.org/abs/1906.08237) [Lample and Conneau (2019)](https://arxiv.org/abs/1901.07291) |
| [**`WordEmbeddings`**](/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md)               | Classic word embeddings                                                                                                                                 |  |
| [**`TransformerDocumentEmbeddings`**](/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md) | Embeddings from pretrained [transformers](https://huggingface.co/transformers/pretrained_models.html) (BERT, XLM, GPT, RoBERTa, XLNet, DistilBERT etc.) | [Devlin et al. (2018)](https://www.aclweb.org/anthology/N19-1423/) [Radford et al. (2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) [Dai et al. (2019)](https://arxiv.org/abs/1901.02860) [Yang et al. (2019)](https://arxiv.org/abs/1906.08237) [Lample and Conneau (2019)](https://arxiv.org/abs/1901.07291) |
| [`DocumentRNNEmbeddings`](/resources/docs/embeddings/DOCUMENT_RNN_EMBEDDINGS.md)            | RNN over word embeddings                                                                                                                                |  |
| [`DocumentPoolEmbeddings`](/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md)          | Pooled word embeddings                                                                                                                                  |  |

You should at least check out the two transformer embeddings. Our recommended order to learn about these embeddings is:
1. Classic word embeddings: [**`WordEmbeddings`**](/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md) 
2. Flair embeddings: [**`FlairEmbeddings`**](/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)
3. Transformer word and document embeddings: [**`TransformerWordEmbeddings`**](/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md) 
4. Any of the others


## Next

Now, let us look at how to use different [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) to embed your
text.
