## Character Embeddings

Some embeddings - such as character-features - are not pre-trained but rather trained on the downstream task. Normally
this requires you to implement a [hierarchical embedding architecture](http://neuroner.com/NeuroNERengine_with_caption_no_figure.png).

With Flair, you don't need to worry about such things. Just choose the appropriate
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