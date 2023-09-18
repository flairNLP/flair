# Classic Word Embeddings

Classic word embeddings are static and word-level, meaning that each distinct word gets exactly one pre-computed
embedding. Most embeddings fall under this class, including the popular GloVe or Komninos embeddings.

Simply instantiate the [`WordEmbeddings`](#flair.embeddings.token.WordEmbeddings) class and pass a string identifier of the embedding you wish to load. So, if
you want to use GloVe embeddings, pass the string 'glove' to the constructor:

```python
from flair.embeddings import WordEmbeddings

# init embedding
glove_embedding = WordEmbeddings('glove')
```
Now, create an example sentence and call the embedding's [`embed()`](#flair.embeddings.base.Embeddings.embed) method. You can also pass a list of sentences to
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
id string to the constructor of the [`WordEmbeddings`](#flair.embeddings.token.WordEmbeddings) class. Typically, you use
the **two-letter language code** to init an embedding, so 'en' for English and
'de' for German and so on. By default, this will initialize FastText embeddings trained over Wikipedia.
You can also always use FastText embeddings over Web crawls, by instantiating with '-crawl'. So 'de-crawl'
to use embeddings trained over German web crawls.

For English, we provide a few more options, so
here you can choose between instantiating 'en-glove', 'en-extvec' and so on.

The following embeddings are currently supported:

| ID | Language | Embedding |
| ------------- | -------------  | ------------- |
| 'en-glove' (or 'glove') | English | GloVe embeddings |
| 'en-extvec' (or 'extvec') | English |Komninos embeddings |
| 'en-crawl' (or 'crawl')  | English | FastText embeddings over Web crawls |
| 'en-twitter' (or 'twitter')  | English | Twitter embeddings |
| 'en-turian' (or 'turian')  | English | Turian embeddings (small) |
| 'en' (or 'en-news' or 'news')  |English | FastText embeddings over news and wikipedia data |
| 'de' | German |German FastText embeddings |
| 'nl' | Dutch | Dutch FastText embeddings |
| 'fr' | French | French FastText embeddings |
| 'it' | Italian | Italian FastText embeddings |
| 'es' | Spanish | Spanish FastText embeddings |
| 'pt' | Portuguese | Portuguese FastText embeddings |
| 'ro' | Romanian | Romanian FastText embeddings |
| 'ca' | Catalan | Catalan FastText embeddings |
| 'sv' | Swedish | Swedish FastText embeddings |
| 'da' | Danish | Danish FastText embeddings |
| 'no' | Norwegian | Norwegian FastText embeddings |
| 'fi' | Finnish | Finnish FastText embeddings |
| 'pl' | Polish | Polish FastText embeddings |
| 'cz' | Czech | Czech FastText embeddings |
| 'sk' | Slovak | Slovak FastText embeddings |
| 'sl' | Slovenian | Slovenian FastText embeddings |
| 'sr' | Serbian | Serbian FastText embeddings |
| 'hr' | Croatian | Croatian FastText embeddings |
| 'bg' | Bulgarian | Bulgarian FastText embeddings |
| 'ru' | Russian | Russian FastText embeddings |
| 'ar' | Arabic | Arabic FastText embeddings |
| 'he' | Hebrew | Hebrew FastText embeddings |
| 'tr' | Turkish | Turkish FastText embeddings |
| 'fa' | Persian | Persian FastText embeddings |
| 'ja' | Japanese | Japanese FastText embeddings |
| 'ko' | Korean | Korean FastText embeddings |
| 'zh' | Chinese | Chinese FastText embeddings |
| 'hi' | Hindi | Hindi FastText embeddings |
| 'id' | Indonesian | Indonesian FastText embeddings |
| 'eu' | Basque | Basque FastText embeddings |

So, if you want to load German FastText embeddings, instantiate as follows:

```python
german_embedding = WordEmbeddings('de')
```

Alternatively, if you want to load German FastText embeddings trained over crawls, instantiate as follows:

```python
german_embedding = WordEmbeddings('de-crawl')
```

We generally recommend the FastText embeddings, or GloVe if you want a smaller model.

If you want to use any other embeddings (not listed in the list above), you can load those by calling
```python
custom_embedding = WordEmbeddings('path/to/your/custom/embeddings.gensim')
```
If you want to load custom embeddings you need to make sure that the custom embeddings are correctly formatted to
[gensim](https://radimrehurek.com/gensim/models/word2vec.html).

You can, for example, convert [FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) to gensim using the
following code snippet:
```python
import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('/path/to/fasttext/embeddings.txt', binary=False)
word_vectors.save('/path/to/converted')
```

However, FastText embeddings have the functionality of returning vectors for out of vocabulary words using the sub-word information. If you want to use this then try [`FastTextEmbeddings`](#flair.embeddings.token.FastTextEmbeddings) class.

