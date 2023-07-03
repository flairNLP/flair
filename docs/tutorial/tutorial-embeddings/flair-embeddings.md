# Flair embeddings

Contextual string embeddings are [powerful embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are:
1) they are trained without any explicit notion of words and thus fundamentally model words as sequences of characters.
2) they are **contextualized** by their surrounding text, meaning that the *same word will have different embeddings depending on its contextual use*.

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

You choose which embeddings you load by passing the appropriate string to the constructor of the [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings) class.
Currently, the following contextual string embeddings are provided (note: replace '*X*' with either '*forward*' or '*backward*'):

| ID                      | Language                                        | Embedding                                                                                                                                                                                                                                    |
|-------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 'multi-X'               | 300+                                            | [JW300 corpus](http://opus.nlpl.eu/JW300.php), as proposed by [Agić and Vulić (2019)](https://www.aclweb.org/anthology/P19-1310/). The corpus is licensed under CC-BY-NC-SA                                                                  
| 'multi-X-fast'          | English, German, French, Italian, Dutch, Polish | Mix of corpora (Web, Wikipedia, Subtitles, News), CPU-friendly                                                                                                                                                                               |
| 'news-X'                | English                                         | Trained with 1 billion word corpus                                                                                                                                                                                                           |
| 'news-X-fast'           | English                                         | Trained with 1 billion word corpus, CPU-friendly                                                                                                                                                                                             |
| 'mix-X'                 | English                                         | Trained with mixed corpus (Web, Wikipedia, Subtitles)                                                                                                                                                                                        |
| 'ar-X'                  | Arabic                                          | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'bg-X'                  | Bulgarian                                       | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'bg-X-fast'             | Bulgarian                                       | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or SETimes)                                                                                                                 |
| 'cs-X'                  | Czech                                           | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'cs-v0-X'               | Czech                                           | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version)                                                                                                                                               |
| 'de-X'                  | German                                          | Trained with mixed corpus (Web, Wikipedia, Subtitles)                                                                                                                                                                                        |
| 'de-historic-ha-X'      | German (historical)                             | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Hamburger Anzeiger*                                                                                                                           |
| 'de-historic-wz-X'      | German (historical)                             | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Wiener Zeitung*                                                                                                                               |
| 'de-historic-rw-X'      | German (historical)                             | Added by [@redewiedergabe](https://github.com/redewiedergabe): Historical German trained over 100 million tokens                                                                                                                             |
| 'es-X'                  | Spanish                                         | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipedia                                                                                                                                             |
| 'es-X-fast'             | Spanish                                         | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipedia, CPU-friendly                                                                                                                               |
| 'es-clinical-'          | Spanish (clinical)                              | Added by [@matirojasg](https://github.com/flairNLP/flair/issues/2292): Trained with Wikipedia                                                                                                                                                |
| 'eu-X'                  | Basque                                          | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'eu-v0-X'               | Basque                                          | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version)                                                                                                                                               |
| 'fa-X'                  | Persian                                         | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'fi-X'                  | Finnish                                         | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'fr-X'                  | French                                          | Added by [@mhham](https://github.com/mhham): Trained with French Wikipedia                                                                                                                                                                   |
| 'he-X'                  | Hebrew                                          | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'hi-X'                  | Hindi                                           | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'hr-X'                  | Croatian                                        | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'id-X'                  | Indonesian                                      | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'it-X'                  | Italian                                         | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'ja-X'                  | Japanese                                        | Added by [@frtacoa](https://github.com/zalandoresearch/flair/issues/527): Trained with 439M words of Japanese Web crawls (2048 hidden states, 2 layers)                                                                                      |
| 'nl-X'                  | Dutch                                           | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'nl-v0-X'               | Dutch                                           | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version)                                                                                                                                               |
| 'no-X'                  | Norwegian                                       | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'pl-X'                  | Polish                                          | Added by [@borchmann](https://github.com/applicaai/poleval-2018): Trained with web crawls (Polish part of CommonCrawl)                                                                                                                       |
| 'pl-opus-X'             | Polish                                          | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'pt-X'                  | Portuguese                                      | Added by [@ericlief](https://github.com/ericlief/language_models): LM embeddings                                                                                                                                                             |
| 'sl-X'                  | Slovenian                                       | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'sl-v0-X'               | Slovenian                                       | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia and OpenSubtitles2018)                                                                                                      |
| 'sv-X'                  | Swedish                                         | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS                                                                                                                                      |
| 'sv-v0-X'               | Swedish                                         | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or OpenSubtitles2018)                                                                                                       |
| 'ta-X'                  | Tamil                                           | Added by [@stefan-it](https://github.com/stefan-it/plur)                                                                                                                                                                                     |
| 'pubmed-X'              | English                                         | Added by [@jessepeng](https://github.com/zalandoresearch/flair/pull/519): Trained with 5% of PubMed abstracts until 2015 (1150 hidden states, 3 layers)                                                                                      |
| 'de-impresso-hipe-v1-X' | German (historical)                             | In-domain data (Swiss and Luxembourgish newspapers) for [CLEF HIPE Shared task](https://impresso.github.io/CLEF-HIPE-2020). More information on the shared task can be found in [this paper](https://zenodo.org/record/3752679#.XqgzxXUzZzU) |
| 'en-impresso-hipe-v1-X' | English (historical)                            | In-domain data (Chronicling America material) for [CLEF HIPE Shared task](https://impresso.github.io/CLEF-HIPE-2020). More information on the shared task can be found in [this paper](https://zenodo.org/record/3752679#.XqgzxXUzZzU)       |
| 'fr-impresso-hipe-v1-X' | French (historical)                             | In-domain data (Swiss and Luxembourgish newspapers) for [CLEF HIPE Shared task](https://impresso.github.io/CLEF-HIPE-2020). More information on the shared task can be found in [this paper](https://zenodo.org/record/3752679#.XqgzxXUzZzU) |
| 'am-X'                  | Amharic                                         | Based on 6.5m Amharic text corpus crawled from different sources. See [this paper](https://www.mdpi.com/1999-5903/13/11/275) and the official [GitHub Repository](https://github.com/uhh-lt/amharicmodels) for more information.             |
| 'uk-X'                  | Ukrainian                                       | Added by [@dchaplinsky](https://github.com/dchaplinsky): Trained with [UberText](https://lang.org.ua/en/corpora/) corpus.                                                                                                                    |

So, if you want to load embeddings from the German forward LM model, instantiate the method as follows:

```python
flair_de_forward = FlairEmbeddings('de-forward')
```

And if you want to load embeddings from the Bulgarian backward LM model, instantiate the method as follows:

```python
flair_bg_backward = FlairEmbeddings('bg-backward')
```

## Recommended Flair usage

We recommend combining both forward and backward Flair embeddings. Depending on the task, we also recommend adding standard [`WordEmbeddings`](#flair.embeddings.token.WordEmbeddings) into the mix. So, our recommended [`StackedEmbeddings`](#flair.embeddings.token.StackedEmbeddings) for most English tasks is:


```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                       ])
```

That's it! Now just use this embedding like all the other embeddings, i.e. call the [`embed()`](#flair.embeddings.base.Embeddings.embed) method over your sentences.

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


## Pooled Flair embeddings

We also developed a pooled variant of the [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings). These embeddings differ in that they *constantly evolve over time*, even at prediction time (i.e. after training is complete). This means that the same words in the same sentence at two different points in time may have different embeddings.

[`PooledFlairEmbeddings`](#flair.embeddings.token.PooledFlairEmbeddings) manage a 'global' representation of each distinct word by using a pooling operation of all past occurences. More details on how this works may be found in [Akbik et al. (2019)](https://www.aclweb.org/anthology/N19-1078/).

You can instantiate and use [`PooledFlairEmbeddings`](#flair.embeddings.token.PooledFlairEmbeddings) like [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings):

```python
from flair.embeddings import PooledFlairEmbeddings

# init embedding
flair_embedding_forward = PooledFlairEmbeddings('news-forward')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
flair_embedding_forward.embed(sentence)
```

Note that while we get some of our best results with [`PooledFlairEmbeddings`](#flair.embeddings.token.PooledFlairEmbeddings) they are very ineffective memory-wise since they keep past embeddings of all words in memory. In many cases, regular [`FlairEmbeddings`](#flair.embeddings.token.FlairEmbeddings) will be nearly as good but with much lower memory requirements.


