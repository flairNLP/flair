# Tutorial 1: NLP Base Types

This is part 1 of the tutorial, in which we look into some of the base types used in this library.

## Creating a Sentence

There are two types of objects that are central to this library, namely the `Sentence` and `Token` objects. A
`Sentence` holds a textual sentence and is essentially a list of `Token`.

Let's start by making a `Sentence` object for an example sentence.

```python
# The sentence objects holds a sentence that we may want to embed or tag
from flair.data import Sentence

# Make a sentence object by passing a string
sentence = Sentence('The grass is green.')

# Print the object to see what's in there
print(sentence)
```

This should print:

```console
Sentence: "The grass is green ."   [− Tokens: 5]
```

The print-out tells us that the sentence consists of 5 tokens.
You can access the tokens of a sentence via their token id or with their index:

```python
# using the token id
print(sentence.get_token(4))
# using the index itself
print(sentence[3])
```

which should print in both cases

```console
Token: 4 green
```

This print-out includes the token id (4) and the lexical value of the token ("green"). You can also iterate over all
tokens in a sentence.

```python
for token in sentence:
    print(token)
```

This should print:

```console
Token: 1 The
Token: 2 grass
Token: 3 is
Token: 4 green
Token: 5 .
```

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

# Print the object to see what's in there
print(untokenized_sentence)
```

In this case, no tokenization is performed and the text is split on whitespaces, thus resulting in only 4 tokens here. 

### Using a different tokenizer

You can also pass custom tokenizers to the initialization method. For instance, if you want to tokenize a Japanese
sentence you can use the 'janome' tokenizer instead, like this: 

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
Sentence: "私 は ベルリン が 好き"   [− Tokens: 5]
```

You can write your own tokenization routine. Check the code of `flair.data.Tokenizer` and its implementations
 (e.g. `flair.tokenization.SegtokTokenizer` or `flair.tokenization.SpacyTokenizer`) to get an idea of how to add 
 your own tokenization method.  

### Using pretokenized sequences
You can alternatively pass a pretokenized sequence as list of words, e.g.

```python
from flair.data import Sentence
sentence = Sentence(['The', 'grass', 'is', 'green', '.'])
print(sentence)
```

This should print:

```console
Sentence: "The grass is green ."   [− Tokens: 5]
```


## Adding Labels

In Flair, any data point can be labeled. For instance, you can label a word or label a sentence:

### Adding labels to tokens

A `Token` has fields for linguistic annotation, such as lemmas, part-of-speech tags or named entity tags. You can
add a tag by specifying the tag type and the tag value. In this example, we're adding an NER tag of type 'color' to
the word 'green'. This means that we've tagged this word as an entity of type color.

```python
# add a tag to a word in the sentence
sentence[3].add_tag('ner', 'color')

# print the sentence with all tags of this type
print(sentence.to_tagged_string())
```

This should print:

```console
The grass is green <color> .
```

Each tag is of class `Label` which next to the value has a score indicating confidence. Print like this: 

```python
# get token 3 in the sentence 
token = sentence[3]

# get the 'ner' tag of the token
tag = token.get_tag('ner')

# print token
print(f'"{token}" is tagged as "{tag.value}" with confidence score "{tag.score}"')
```

This should print:

```console
"Token: 4 green" is tagged as "color" with confidence score "1.0"
```

Our color tag has a score of 1.0 since we manually added it. If a tag is predicted by our
sequence labeler, the score value will indicate classifier confidence.

### Adding labels to sentences

You can also add a `Label` to a whole `Sentence`.
For instance, the example below shows how we add the label 'sports' to a sentence, thereby labeling it
as belonging to the sports "topic".

```python
sentence = Sentence('France is the current world cup winner.')

# add a label to a sentence
sentence.add_label('topic', 'sports')

print(sentence)

# Alternatively, you can also create a sentence with label in one line
sentence = Sentence('France is the current world cup winner.').add_label('topic', 'sports')

print(sentence)
```

This should print: 

```console
Sentence: "France is the current world cup winner."   [− Tokens: 7  − Sentence-Labels: {'topic': [sports (1.0)]}]
```

Indicating that this sentence belongs to the topic 'sports' with confidence 1.0.

### Multiple labels

Any data point can be labeled multiple times. A sentence for instance might belong to two topics. In this case, add two labels with the same label name:

```python
sentence = Sentence('France is the current world cup winner.')

# this sentence has multiple topic labels
sentence.add_label('topic', 'sports')
sentence.add_label('topic', 'soccer')
```

You might want to add different layers of annotation for the same sentence. Next to topic you might also want to predict the "language" of a sentence. In this case, add a label with a different label name: 

```python
sentence = Sentence('France is the current world cup winner.')

# this sentence has multiple "topic" labels
sentence.add_label('topic', 'sports')
sentence.add_label('topic', 'soccer')

# this sentence has a "language" label
sentence.add_label('language', 'English')

print(sentence)
```

This should print: 

```console
Sentence: "France is the current world cup winner."   [− Tokens: 7  − Sentence-Labels: {'topic': [sports (1.0), soccer (1.0)], 'language': [English (1.0)]}]
```

Indicating that this sentence has two "topic" labels and one "language" label. 

### Accessing a sentence's labels

You can access these labels like this: 

```python
for label in sentence.labels:
    print(label)
```

Remember that each label is a `Label` object, so you can also access the label's `value` and `score` fields directly:

```python
print(sentence.to_plain_string())
for label in sentence.labels:
    print(f' - classified as "{label.value}" with score {label.score}')
```

This should print:

```console
France is the current world cup winner.
 - classified as "sports" with score 1.0
 - classified as "soccer" with score 1.0
 - classified as "English" with score 1.0
```

If you are interested only in the labels of one layer of annotation, you can access them like this: 

```python
for label in sentence.get_labels('topic'):
    print(label)
```

Giving you only the "topic" labels.


## Next

So far, we've seen how to create sentences and label them manually.

Now, let us look at how to use [pre-trained models](/resources/docs/TUTORIAL_2_TAGGING.md) to tag your text.
