# Basics

This tutorial explains the basic concepts used in Flair:

-    what is a [`Sentence`](#flair.data.Sentence)
-    what is a [`Label`](#flair.data.Label)

You should be familiar with these two concepts in order to get the most out of Flair.

## What is a Sentence

If you want to tag a sentence, you need to first make a [`Sentence`](#flair.data.Sentence) object for it.

For example, say you want to tag the text "_The grass is green._".

Let's start by making a [`Sentence`](#flair.data.Sentence) object for this sentence.


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
Sentence[5]: "The grass is green."
```

The print-out tells us that the sentence consists of 5 tokens.

```{note}
A token is an atomic unit of the text, often a word or punctuation. The printout is therefore telling us that the sentence "_The grass is green._" consists of 5 such atomic units. 
```

### Iterating over the tokens in a Sentence

So what are the 5 tokens in this example sentence?

You can iterate over all tokens in a sentence like this:


```python
for token in sentence:
    print(token)
```

This should print:

```console
Token[0]: "The"
Token[1]: "grass"
Token[2]: "is"
Token[3]: "green"
Token[4]: "."
```

This printout is telling us that the 5 tokens in the text are the words "_The_", "_grass_", "_is_", "_green_", with a separate token for the full stop at the end. The tokens therefore correspond to the words and the punctuation of the text.

### Directly accessing a token

You can access the tokens of a sentence via their token id or with their index:

```python
# using the token id
print(sentence.get_token(4))
# using the index itself
print(sentence[3])
```

which should print in both cases

```console
Token[3]: "green"
```

This print-out includes the token index (3) and the lexical value of the token ("green"). 

### Tokenization

When you create a [`Sentence`](#flair.data.Sentence) as above, the text is automatically tokenized (segmented into words) using the [segtok](https://pypi.org/project/segtok/) library.

```{note}
You can also use a different tokenizer if you like. To learn more about this, check out our tokenization tutorial.
```


## What is a Label

All Flair models predict labels. For instance, our sentiment analysis models will predict labels for a sentence. Our NER models will predict labels for tokens in a sentence.

### Example 1: Labeling a token in a sentence

To illustrate how labels work, let's use the same example sentence as above: "_The grass is green._".

Let us label all "color words" in this sentence. Since the sentence contains only one color word (namely "green"), we only need to add a label to one of the tokens.

We access token 3 in the sentence, and set a label for it: 

```python
# Make a sentence object by passing a string
sentence = Sentence('The grass is green.')

# add an NER tag to token 3 in the sentence
sentence[3].add_label('ner', 'color')

# print the sentence (now with this annotation)
print(sentence)
```

This should print:

```console
Sentence: "The grass is green ." → ["green"/color]
```

The output indicates that the word "green" in this sentence is labeled as a "color". You can also
iterate through each token and print it to see if it has labels:

```python
for token in sentence:
    print(token)
```

This should print:

```console
Token[0]: "The"
Token[1]: "grass"
Token[2]: "is"
Token[3]: "green" → color (1.0)
Token[4]: "."
```

This shows that there are 5 tokens in the sentence, one of which has a label.

```{note}
The [`add_label`](#flair.data.DataPoint.add_label) method used here has two mandatory parameters.
```

### Example 2: Labeling a whole sentence

Sometimes you want to label an entire sentence instead of only a token. Do this by calling [`add_label`](#flair.data.DataPoint.add_label) for the whole sentence.

For example, say we want to add a sentiment label to the sentence "_The grass is green._":

```python
sentence = Sentence('The grass is green.')

# add a label to a sentence
sentence.add_label('sentiment', 'POSITIVE')

print(sentence)
```

This should print:

```
Sentence[5]: "The grass is green." → POSITIVE (1.0)
```

Indicating that this sentence is now labeled as having a positive sentiment.

### Multiple labels

Importantly, in Flair you can add as many labels to a sentence as you like.

Let's bring the two examples above together: We will label the sentence "_The grass is green._" with an overall positive sentiment, and also add a "color" tag to the token "grass":

```python
sentence = Sentence('The grass is green.')

# add a sentiment label to the sentence
sentence.add_label('sentiment', 'POSITIVE')

# add an NER tag to token 3 in the sentence
sentence[3].add_label('ner', 'color')

# print the sentence with all annotations
print(sentence)
```

This will print:

```
Sentence[5]: "The grass is green." → POSITIVE (1.0) → ["green"/color]
```

Indicating that the sentence is now labeled with two different types of information.

### Accessing labels

You can iterate through all labels of a sentence using the [`get_labels()`](#flair.data.Sentence.get_labels) method:

```python
# iterate over all labels and print
for label in sentence.get_labels():
    print(label)
```

This will get each label and print it. For instance, let's re-use the previous example in which we add two different labels to the same sentence:

```python
sentence = Sentence('The grass is green.')

# add a sentiment label to the sentence
sentence.add_label('sentiment', 'POSITIVE')

# add an NER tag to token 3 in the sentence
sentence[3].add_label('ner', 'color')

# iterate over all labels and print
for label in sentence.get_labels():
    print(label)
```

This will now print the following two lines:

```
Sentence[5]: "The grass is green." → POSITIVE (1.0)
Token[3]: "green" → color (1.0)
```

This printout tells us that there are two labels: The first is for the whole sentence, tagged as POSITIVE. The second is only for the token "green", tagged as "color".

````{note}

If you only want to iterate over labels of a specific type, add the label name as parameter to [`get_labels()`](#flair.data.Sentence.get_labels). For instance, to only iterate over all NER labels, do:

```python
# iterate over all NER labels only
for label in sentence.get_labels('ner'):
    print(label)
```
````

### Information for each label

Each label is of class `Label` which next to the value has a score indicating confidence. It also has a pointer back to the data point to which it attaches.

This means that you can print the value, the confidence and the labeled text of each label:

```python
sentence = Sentence('The grass is green.')

# add an NER tag to token 3 in the sentence
sentence[3].add_label('ner', 'color')

# iterate over all labels and print
for label in sentence.get_labels():

    # Print the text, the label value and the label score
    print(f'"{label.data_point.text}" is classified as "{label.value}" with score {label.score}')
```

This should print:

```
"green" is classified as "color" with score 1.0
```

Our color tag has a score of 1.0 since we manually added it. If a tag is predicted by our sequence labeler, the score value will indicate classifier confidence.

