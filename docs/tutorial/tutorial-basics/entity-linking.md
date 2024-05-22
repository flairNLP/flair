# Tagging and linking entities

As of Flair 0.12 we ship an **experimental entity linker** trained on the [Zelda dataset](https://github.com/flairNLP/zelda). The linker does not only
tag entities, but also attempts to link each entity to the corresponding Wikipedia URL if one exists. 

## Example 1: Entity linking on a single sentence​

To illustrate, let's use the example sentence "_Kirk and Spock met on the Enterprise._":

```python
from flair.nn import Classifier
from flair.data import Sentence

# load the model
tagger = Classifier.load('linker')

# make a sentence
sentence = Sentence('Kirk and Spock met on the Enterprise.')

# predict entity links
tagger.predict(sentence)

# iterate over predicted entities and print
for label in sentence.get_labels():
    print(label)
```

This should print:
```console
Span[0:1]: "Kirk" → James_T._Kirk (0.9969)
Span[2:3]: "Spock" → Spock (0.9971)
Span[6:7]: "Enterprise" → USS_Enterprise_(NCC-1701-D) (0.975)
```

As we can see, the linker can resolve what the two mentions of "Barcelona" refer to: 
- "Kirk" refers to the entity "[James_T._Kirk](https://en.wikipedia.org/wiki/James_T._Kirk)"
- "Spock" refers to "[Spock](https://en.wikipedia.org/wiki/Spock)" (ok, that one was easy)
- "Enterprise" refers to the "[USS_Enterprise_(NCC-1701-D)](https://en.wikipedia.org/wiki/USS_Enterprise_(NCC-1701-D))" 

 Not bad, eh? However, that last prediction is not quite correct as Star Trek fans will know. Entity linking is a hard task and we are working to improve the accuracy of our model.



## Example 2: Entity linking on a text document (multiple sentences)

Entity linking typically works best when applied to a whole document instead of only a single sentence.

To illustrate how this works, let's use the following short text: "_Bayern played against Barcelona. The match took place in Barcelona._"

In this case, split the text into sentences and pass a list of Sentence objects to the [`Classifier.predict()`](#flair.nn.Classifier.predict) method:

```python
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# example text with many sentences
text = "Bayern played against Barcelona. The match took place in Barcelona."

# initialize sentence splitter
splitter = SegtokSentenceSplitter()

# use splitter to split text into list of sentences
sentences = splitter.split(text)

# predict tags for sentences
tagger = Classifier.load('linker')
tagger.predict(sentences)

# iterate through sentences and print predicted labels
for sentence in sentences:
    print(sentence)
```

This should print: 
```console
Sentence[5]: "Bayern played against Barcelona." → ["Bayern"/FC_Bayern_Munich, "Barcelona"/FC_Barcelona]
Sentence[7]: "The match took place in Barcelona." → ["Barcelona"/Barcelona]
```

As we can see, the linker can resolve that:

- "Bayern" refers to the soccer club "[FC Bayern Munich](https://en.wikipedia.org/wiki/FC_Bayern_Munich)"
- the first mention of "Barcelona" refers to the soccer club "[FC Barcelona](https://en.wikipedia.org/wiki/FC_Barcelona)"
- the second mention of "Barcelona" refers to the city of "[Barcelona](https://en.wikipedia.org/wiki/Barcelona)"

