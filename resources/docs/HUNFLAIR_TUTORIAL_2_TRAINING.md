# HunFlair Tutorial 2: Training NER models

This part of the tutorial shows how you can train your own biomedical named entity recognition models 
using state-of-the-art word embeddings.

For this tutorial, we assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of Flair
and how [word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md) and 
[flair embeddings](/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md) work. 
You should also know how to [load a corpus](/resources/docs/TUTORIAL_6_CORPUS.md).

## Train a biomedical NER model from scratch
Here is example code for a biomedical NER model trained over `NCBI_DISEASE` corpus, using word embeddings 
and flair embeddings based on biomedical abstracts from PubMed and full-texts from PMC.
```python
from flair.datasets import NCBI_DISEASE

# 1. get the corpus
corpus = NCBI_DISEASE()
print(corpus)

# 2. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type="ner")

# 3. initialize embeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

embedding_types = [

    # word embeddings trained on PubMed and PMC
    WordEmbeddings("pubmed"),

    # flair embeddings trained on PubMed and PMC
    FlairEmbeddings("pubmed-forward"),
    FlairEmbeddings("pubmed-backward"),
]


embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 4. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True,
    locked_dropout=0.5
)

# 5. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(
    base_path="taggers/ncbi-disease",
    train_with_dev=False,
    max_epochs=200,
    learning_rate=0.1,
    mini_batch_size=32
)
```
Once the model is trained you can use it to predict tags for new sentences. 
Just call the predict method of the model.
```python
# load the model you trained
model = SequenceTagger.load("taggers/ncbi-disease/best-model.pt")

# create example sentence
from flair.data import Sentence
sentence = Sentence("Women who smoke 20 cigarettes a day are four times more likely to develop breast cancer.")

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
```
If the model works well, it will correctly tag "breast cancer" as disease in this example:
~~~
Women who smoke 20 cigarettes a day are four times more likely to develop breast <B-Disease> cancer <E-Disease> .
~~~

## Fine-tuning HunFlair models 
Next to training a model completely from scratch, there is also the opportunity to just fine-tune 
the *HunFlair* models (or any other pre-trained model) to your target domain / corpus. 
This can be advantageous because the pre-trained models are based on a much broader data base, 
which may allows a better and faster adaptation to the target domain. In the following example
we fine-tune the `hunflar-disease` model to the `NCBI_DISEASE`:   
```python
# 1. load your target corpus
from flair.datasets import NCBI_DISEASE
corpus = NCBI_DISEASE()

# 2. load the pre-trained sequence tagger
from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger.load("hunflair-disease")

# 3. initialize trainer
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 4. fine-tune on the target corpus
trainer.train(
    base_path="taggers/hunflair-disease-finetuned-ncbi",
    train_with_dev=False,
    max_epochs=200,
    learning_rate=0.1,
    mini_batch_size=32
)
```
## Training HunFlair from scratch
*HunFlair* consists of distinct models for the entity types cell line, chemical, disease, gene/protein
and species. For each entity multiple corpora are used to train the model for the specific entity. The 
following code examples illustrates the training process of *HunFlair* for *cell line*:

```python
from flair.datasets import HUNER_CELL_LINE

# 1. get all corpora for a specific entity type
from flair.models import SequenceTagger
corpus = HUNER_CELL_LINE()

# 2. initialize embeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
embedding_types = [
    WordEmbeddings("pubmed"),
    FlairEmbeddings("pubmed-forward"),
    FlairEmbeddings("pubmed-backward"),

]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 3. initialize sequence tagger
tag_dictionary = corpus.make_label_dictionary(label_type="ner")

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True,
    locked_dropout=0.5
)

# 4. train the model
from flair.trainers import ModelTrainer
trainer = ModelTrainer(tagger, corpus)

trainer.train(
    base_path="taggers/hunflair-cell-line", 
    train_with_dev=False, 
    max_epochs=200,
    learning_rate=0.1, 
    mini_batch_size=32
)
```
Analogously, distinct models can be trained for chemicals, diseases, genes/proteins and species using 
`HUNER_CHEMICALS`, `HUNER_DISEASE`, `HUNER_GENE`, `HUNER_SPECIES` respectively. 


