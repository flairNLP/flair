# Character Embeddings

`CharacterEmbeddings` allow you to add character-level word embeddings during model training. Note that these embeddings
are randomly initialized when you initialize the class, so they are not meaningful unless you train them on a specific
downstream task.

For instance, the standard sequence labeling architecture used by [Lample et al. (2016)](https://www.aclweb.org/anthology/N16-1030) is a combination of classic word embeddings with task-trained character features. Normally this would require you to implement a [hierarchical embedding architecture](http://neuroner.com/NeuroNERengine_with_caption_no_figure.png) in which character-level embeddings for each word are computed using an RNN and then concatenated with word embeddings.

In Flair, we simplify this by treating `CharacterEmbeddings` just like any other embedding class. To reproduce the
Lample architecture, you need only combine them with standard `WordEmbeddings` in an embedding stack:


```python
# init embedding stack
embedding = StackedEmbeddings(
    [
        # standard word embeddings
        WordEmbeddings('glove'),

        # character-level features
        CharacterEmbeddings(),
    ]
)
```

If you pass this stacked embedding to a train method, the character-level features will now automatically be trained
for your downstream task.
