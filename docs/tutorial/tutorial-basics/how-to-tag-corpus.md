# How to tag a whole corpus

Often, you may want to tag an entire text corpus. In this case, you need to split the corpus into sentences and pass a
list of [`Sentence`](#flair.data.Sentence) objects to the [`Classifier.predict()`](#flair.nn.Classifier.predict) method.

For instance, you can use a [`SentenceSplitter`](#flair.splitter.SentenceSplitter) to split your text:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "This is a sentence. This is another sentence. I love Berlin."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = Classifier.load('ner')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

Using the `mini_batch_size` parameter of the [`Classifier.predict()`](#flair.nn.Classifier.predict) method, you can set the size of mini batches passed to the
tagger. Depending on your resources, you might want to play around with this parameter to optimize speed.

