# Tutorial 1.1: Tokenization in Flair

Tokenization is used to detect words in a given text. 
This tutorial gives more details on tokenization in Flair and how to change tokenizers. 

## Example Sentence

Let us use the following example sentence: "The grass is green." 

To show default tokenization, we make a `Sentence` object and iterate over all tokens: 

```python
# The sentence objects holds a sentence that we may want to embed or tag
from flair.data import Sentence

# Make a sentence object by passing a string
sentence = Sentence('The grass is green.')

# Iterate over each token in sentence and print
for token in sentence:
    print(token)
```

This will print: 

```console
Token[0]: "The"
Token[1]: "grass"
Token[2]: "is"
Token[3]: "green"
Token[4]: "."
```

Showing us that 5 tokens are automatically detected. 

## Tokenization

When you create a `Sentence` as above, the text is **automatically tokenized** using the
lightweight [segtok library](https://pypi.org/project/segtok/).

### Using no tokenizer

If you *do not* want to use this tokenizer, simply set the `use_tokenizer` flag to `False`
when instantiating your `Sentence` with an untokenized string:

```python
from flair.data import Sentence

# Make a sentence object by passing an untokenized string and the 'use_tokenizer' flag
untokenized_sentence = Sentence('The grass is green.', use_tokenizer=False)

# Print the sentence
print(untokenized_sentence)

# Print the number of tokens in sentence (only 4 because no tokenizer is used)
print(len(untokenized_sentence))
```

In this case, no tokenization is performed and the text is split on whitespaces, thus resulting in only 4 tokens here.

### Using a different tokenizer

You can also pass custom tokenizers to the initialization method. For instance, you can tokenize Japanese sentences with the konoha tokenizer. For this, you first need to install konoha. 

```
pip install konoha
```

And then run this code: 

```python
from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer

# init japanese tokenizer
tokenizer = JapaneseTokenizer("janome")

# make sentence (and tokenize)
japanese_sentence = Sentence("私はベルリンが好き", use_tokenizer=tokenizer)

# output tokenized sentence
print(japanese_sentence)
```

This should print:

```console
Sentence[5]: "私はベルリンが好き"
```

Similarly, you can use a tokenizer provided by SpaCy. 


### Using pretokenized sequences
You can alternatively pass a pretokenized sequence as list of words, e.g.

```python
from flair.data import Sentence
sentence = Sentence(['The', 'grass', 'is', 'green', '.'])
print(sentence)
```

This should print:

```console
Sentence[5]: "The grass is green ."
```

