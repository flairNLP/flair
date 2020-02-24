# ELMo Embeddings

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

By default, the top 3 layers in the language model are concatenated to form the word embedding.

AllenNLP provides the following pre-trained models. To use any of the following models inside Flair
simple specify the embedding id when initializing the `ELMoEmbeddings`.

| ID | Language | Embedding |
| ------------- | ------------- | ------------- |
| 'small' | English | 1024-hidden, 1 layer, 14.6M parameters |
| 'medium'   | English | 2048-hidden, 1 layer, 28.0M parameters |
| 'original'    | English | 4096-hidden, 2 layers, 93.6M parameters |
| 'large'    | English |  |
| 'pt'   | Portuguese | |
| 'pubmed' | English biomedical data | [more information](https://allennlp.org/elmo) |
