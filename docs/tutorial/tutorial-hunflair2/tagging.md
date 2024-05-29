# HunFlair2 - Tutorial 1: Tagging

This is part 1 of the tutorial, in which we show how to use our pre-trained *HunFlair2* models to tag your text.

## Tagging with Pre-trained HunFlair2-Models
Let's use the pre-trained *HunFlair2* model for biomedical named entity recognition (NER).
This model was trained over multiple biomedical NER data sets and can recognize 5 different entity types,
i.e. cell lines, chemicals, disease, gene / proteins and species.
```python
from flair.nn import Classifier

tagger = Classifier.load("hunflair2")
```
All you need to do is use the predict() method of the tagger on a sentence.
This will add predicted tags to the tokens in the sentence.
Lets use a sentence with four named entities:
```python
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

# predict NER tags
tagger.predict(sentence)

# print the predicted tags
for entity in sentence.get_labels():
    print(entity)
```
This should print:
```console
Span[0:2]: "Behavioral abnormalities" → Disease (1.0)
Span[4:5]: "Fmr1" → Gene (1.0)
Span[6:7]: "Mouse" → Species (1.0)
Span[9:12]: "Fragile X Syndrome" → Disease (1.0)
```
The output indicates that there are two diseases mentioned in the text ("_Behavioral Abnormalities_" and 
"_Fragile X Syndrome_") as well as one gene ("_fmr1_") and one species ("_Mouse_"). For each entity the
text span in the sentence mention it is given and Label with a value and a score (confidence in the 
prediction). You can also get additional information, such as the position offsets of each entity 
in the sentence in a structured way by calling the `to_dict()` method:

```python
print(sentence.to_dict())
```
This should print:
```python
{
    'text': 'Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome', 
    'labels': [], 
    'entities': [
        {'text': 'Behavioral abnormalities', 'start_pos': 0, 'end_pos': 24, 'labels': [{'value': 'Disease', 'confidence': 0.9999860525131226}]}, 
        {'text': 'Fmr1', 'start_pos': 32, 'end_pos': 36, 'labels': [{'value': 'Gene', 'confidence': 0.9999895095825195}]}, 
        {'text': 'Mouse', 'start_pos': 41, 'end_pos': 46, 'labels': [{'value': 'Species', 'confidence': 0.9999873638153076}]}, 
        {'text': 'Fragile X Syndrome', 'start_pos': 56, 'end_pos': 74, 'labels': [{'value': 'Disease', 'confidence': 0.9999928871790568}]}
      ],
    # further sentence information
}
```

## Using a Biomedical Tokenizer
Tokenization, i.e. separating a text into tokens / words, is an important issue in natural language processing
in general and biomedical text mining in particular. So far, we used a tokenizer for general domain text.
This can be unfavourable if applied to biomedical texts.

*HunFlair2* integrates [SciSpaCy](https://allenai.github.io/scispacy/), a library specially designed to work with scientific text.
To use the library we first have to install it and download one of its models:
~~~
pip install scispacy==0.5.1
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
~~~

Then we can use the [`SciSpacyTokenizer`](#flair.tokenization.SciSpacyTokenizer), we just have to pass it as parameter to when instancing a sentence:
```python
from flair.tokenization import SciSpacyTokenizer

tokenizer = SciSpacyTokenizer()

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome",
                    use_tokenizer=tokenizer)
```

## Working with longer Texts
Often, we are concerned with complete scientific abstracts or full-texts when performing biomedical text mining, e.g.
```python
abstract = "Fragile X syndrome (FXS) is a developmental disorder caused by a mutation in the X-linked FMR1 gene, " \
           "coding for the FMRP protein which is largely involved in synaptic function. FXS patients present several " \
           "behavioral abnormalities, including hyperactivity, anxiety, sensory hyper-responsiveness, and cognitive " \
           "deficits. Autistic symptoms, e.g., altered social interaction and communication, are also often observed: " \
           "FXS is indeed the most common monogenic cause of autism."
```

To work with complete abstracts or full-text, we first have to split them into separate sentences.
We can apply the [`SciSpacySentenceSplitter`](#flair.splitter.SciSpacySentenceSplitter), an integration of the [SciSpaCy](https://allenai.github.io/scispacy/) library:
```python
from flair.splitter import SciSpacySentenceSplitter

# initialize the sentence splitter
splitter = SciSpacySentenceSplitter()

# split text into a list of Sentence objects
sentences = splitter.split(abstract)

# you can apply the HunFlair tagger directly to this list
tagger.predict(sentences)
```
We can access the annotations of the single sentences by just iterating over the list:
```python
for sentence in sentences:
    print(sentence.to_tagged_string())
```
This should print:
~~~
Sentence[35]: "Fragile X syndrome (FXS) is a developmental disorder caused by a mutation in the X-linked FMR1 gene, coding for the FMRP protein which is largely involved in synaptic function." \
              → ["Fragile X syndrome"/Disease, "FXS"/Disease, "developmental disorder"/Disease, "X-linked"/Gene, "FMR1"/Gene, "FMRP"/Gene]
Sentence[23]: "FXS patients present several behavioral abnormalities, including hyperactivity, anxiety, sensory hyper-responsiveness, and cognitive deficits." \
              → ["FXS"/Disease, "patients"/Species, "behavioral abnormalities"/Disease, "hyperactivity"/Disease, "anxiety"/Disease, "sensory hyper-responsiveness"/Disease, "cognitive deficits"/Disease]
Sentence[27]: "Autistic symptoms, e.g., altered social interaction and communication, are also often observed: FXS is indeed the most common monogenic cause of autism." \
              → ["Autistic symptoms"/Disease, "altered social interaction and communication"/Disease, "FXS"/Disease, "autism"/Disease]
~~~
