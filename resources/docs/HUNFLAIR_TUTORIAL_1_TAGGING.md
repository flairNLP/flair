# HunFlair Tutorial 1: Tagging

This is part 1 of the tutorial, in which we show how to use our pre-trained *HunFlair* models to tag your text.

### Tagging with Pre-trained HunFlair-Models
Let's use the pre-trained *HunFlair* model for biomedical named entity recognition (NER). 
This model was trained over 24 biomedical NER data sets and can recognize 5 different entity types,
i.e. cell lines, chemicals, disease, gene / proteins and species.
```python
from flair.models import MultiTagger

tagger = MultiTagger.load("hunflair")
```
All you need to do is use the predict() method of the tagger on a sentence. 
This will add predicted tags to the tokens in the sentence. 
Lets use a sentence with four named entities:
```python
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())
```
This should print:
~~~
Sentence: "Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome" → ["Behavioral abnormalities"/Disease, "Fmr1"/Gene, "Mouse"/Species, "Fragile X Syndrome"/Disease]
~~~
The output contains the words of the original text extended by tags indicating whether
the word is the beginning (B), inside (I) or end (E) of an entity. 
For example, "Fragil" is the first word of the disease "Fragil X Syndrom".
Entities consisting of just one word are marked with a special single tag (S). 
For example, "Mouse" refers to a species entity. 

### Getting Annotated Spans
Often named entities consist of multiple words spanning a certain text span in the input text, such as 
"_Behavioral Abnormalities_" or "_Fragile X Syndrome_" in our example sentence. 
You can directly get such spans in a tagged sentence like this:
```python
for disease in sentence.get_spans("hunflair-disease"):
    print(disease)
```
This should print:
~~~
Span[0:2]: "Behavioral abnormalities" → Disease (0.6736)
Span[9:12]: "Fragile X Syndrome" → Disease (0.99)
~~~

Which indicates that "_Behavioral Abnormalities_" or "_Fragile X Syndrome_" are both disease. 
Each such Span has a text, its position in the sentence and Label with a value and a score 
(confidence in the prediction). You can also get additional information, such as the position 
offsets of each entity in the sentence by calling the `to_dict()` method:
```python
print(sentence.to_dict("hunflair-disease"))
```
This should print:
~~~
{
    'text': 'Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome',
    'hunflair-disease': [
        {'value': 'Disease', 'confidence': 0.6735622882843018},
        {'value': 'Disease', 'confidence': 0.9900058706601461}
    ]
}
~~~

You can retrieve all annotated entities of the other entity types in analogous way using `hunflair-cellline`
for cell lines,  `hunflair-chemical` for chemicals, `hunflair-gene` for genes and proteins, and `hunflair-species`
for species. To get all entities in one you can run:
```python
for annotation_layer in sentence.annotation_layers.keys():
    for entity in sentence.get_spans(annotation_layer):
        print(entity)
```   
This should print:
~~~
Span[0:2]: "Behavioral abnormalities" → Disease (0.6736)
Span[9:12]: "Fragile X Syndrome" → Disease (0.99)
Span[4:5]: "Fmr1" → Gene (0.838)
Span[6:7]: "Mouse" → Species (0.9979)
~~~

### Using a Biomedical Tokenizer
Tokenization, i.e. separating a text into tokens / words, is an important issue in natural language processing 
in general and biomedical text mining in particular. So far, we used a tokenizer for general domain text. 
This can be unfavourable if applied to biomedical texts. 

*HunFlair* integrates [SciSpaCy](https://allenai.github.io/scispacy/), a library specially designed to work with scientific text. 
To use the library we first have to install it and download one of it's models:
~~~
pip install scispacy==0.2.5
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
~~~

To use the tokenizer we just have to pass it as parameter to when instancing a sentence:
```python
from flair.tokenization import SciSpacyTokenizer

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome",  
                    use_tokenizer=SciSpacyTokenizer())
```

### Working with longer Texts
Often, we are concerned with complete scientific abstracts or full-texts when performing
biomedical text mining, e.g. 
```python
abstract = "Fragile X syndrome (FXS) is a developmental disorder caused by a mutation in the X-linked FMR1 gene, " \
           "coding for the FMRP protein which is largely involved in synaptic function. FXS patients present several " \
           "behavioral abnormalities, including hyperactivity, anxiety, sensory hyper-responsiveness, and cognitive " \
           "deficits. Autistic symptoms, e.g., altered social interaction and communication, are also often observed: " \
           "FXS is indeed the most common monogenic cause of autism."
```

To work with complete abstracts or full-text, we first have to split them into separate sentences.
Again we can apply the integration of the [SciSpaCy](https://allenai.github.io/scispacy/) library:
```python
from flair.tokenization import SciSpacySentenceSplitter

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
Sentence: "Fragile X syndrome ( FXS ) is a developmental disorder caused by a mutation in the X - linked FMR1 gene , coding for the FMRP protein which is largely involved in synaptic function ." → ["Fragile X syndrome"/Disease, "FXS"/Disease, "developmental disorder"/Disease, "FMR1"/Gene, "FMRP"/Gene]
Sentence: "FXS patients present several behavioral abnormalities , including hyperactivity , anxiety , sensory hyper - responsiveness , and cognitive deficits ." → ["FXS"/Disease, "behavioral abnormalities"/Disease, "hyperactivity"/Disease, "anxiety"/Disease, "cognitive deficits"/Disease]
Sentence: "Autistic symptoms , e.g. , altered social interaction and communication , are also often observed : FXS is indeed the most common monogenic cause of autism ." → ["Autistic symptoms"/Disease, "FXS"/Disease, "autism"/Disease]
~~~

### Next
Now, let us look at how to [train your own biomedical models](HUNFLAIR_TUTORIAL_2_TRAINING.md) to tag your text.


 
