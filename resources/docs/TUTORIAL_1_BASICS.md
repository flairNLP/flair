# Tutorial 1: NLP Base Types

This is part 1 of the tutorial, in which we look into some of the base types used in this library.

## Creating a Sentence

There are two types of objects that are central to this library, namely the `Sentence` and `Token` objects. A
`Sentence` holds a textual sentence and is essentially a list of `Token`.

Let's start by making a `Sentence` object for an example sentence.

```python
# The sentence objects holds a sentence that we may want to embed or tag
from flair.data import Sentence

# Make a sentence object by passing a whitespace tokenized string
sentence = Sentence('The grass is green .')

# Print the object to see what's in there
print(sentence)
```

This should print:

```console
Sentence: "The grass is green ." - 5 Tokens
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

In some use cases, you might not have your text already tokenized. For this case, we added a simple tokenizer using the
lightweight [segtok library](https://pypi.org/project/segtok/).

Simply use the `use_tokenizer` flag when instantiating your `Sentence` with an untokenized string:

```python
from flair.data import Sentence

# Make a sentence object by passing an untokenized string and the 'use_tokenizer' flag
sentence = Sentence('The grass is green.', use_tokenizer=True)

# Print the object to see what's in there
print(sentence)
```

This should print:

```console
Sentence: "The grass is green ." - 5 Tokens
```

## Adding Tags to Tokens

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
from flair.data import Label

tag: Label = sentence[3].get_tag('ner')

print(f'"{sentence[3]}" is tagged as "{tag.value}" with confidence score "{tag.score}"')
```

This should print:

```console
"Token: 4 green" is tagged as "color" with confidence score "1.0"
```

Our color tag has a score of 1.0 since we manually added it. If a tag is predicted by our
sequence labeler, the score value will indicate classifier confidence.

## Adding Labels to Sentences

A `Sentence` can have one or multiple labels that can for example be used in text classification tasks.
For instance, the example below shows how we add the label 'sports' to a sentence, thereby labeling it
as belonging to the sports category.

```python
sentence = Sentence('France is the current world cup winner.')

# add a label to a sentence
sentence.add_label('sports')

# a sentence can also belong to multiple classes
sentence.add_labels(['sports', 'world cup'])

# you can also set the labels while initializing the sentence
sentence = Sentence('France is the current world cup winner.', labels=['sports', 'world cup'])
```

Labels are also of the `Label` class. So, you can print a sentence's labels like this: 

```python
sentence = Sentence('France is the current world cup winner.', labels=['sports', 'world cup'])

print(sentence)
for label in sentence.labels:
    print(label)
```

This should print:

```console
sports (1.0)
world cup (1.0)
```

This indicates that the sentence belongs to these two classes, each with confidence score 1.0.

## Next

Now, let us look at how to use [pre-trained models](/resources/docs/TUTORIAL_2_TAGGING.md) to tag your text.
