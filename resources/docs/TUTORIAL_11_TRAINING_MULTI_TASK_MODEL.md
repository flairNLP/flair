# BETA - Tutorial 11: Multitask Learning

Transfer learning refers to the situation where one wants to exploit
knowledge obtained in one setting to improve generalization in another
setting. The assumption why transfer learning and its variants
(multitask learning, sequential transfer learning, ...) may generalize
better in another setting is simple: similar tasks share low-level features.
In NLP, different tasks might share the same underlying concept of
grammatical structure of a language.
    
## Multitask Learning in flair

flair comes with a modular multitask learning framework that you can
easily configure your very own multitask learning model. In order to 
create the multitask learning models you want, it is important to understand
how multitask models are built in flair: bottom-up. This means,
you will still create one model per task you want to accomplish, but
if you want to share certain layers across tasks you need to create
them in advance.

Since this development is still in progress, we are currently
supporting (not stable):
- Combine sequence labeling and text classification tasks
- Share different embeddings across multiple tasks

What we currently still missing:
- A proper logging
- Custom loss averaging functions across multiple tasks
- Usw Few-Shot Classification or Language Modeling as a task

This features will come but depend on some refactorings which we need
to implement first.

## Example 1: Two or more Sequence Labeling tasks
```python
import flair
from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CONLL_03, WNUT_17
from flair.embeddings import TransformerWordEmbeddings
from flair.models import MultitaskModel, SequenceTagger
from flair.trainers import ModelTrainer
from transformers import AutoTokenizer, AutoModel, AutoConfig

# use GPU if available
flair.device = "cuda:0"
task = 'ner'

# ----- CORPORA -----
conll03: Corpus = CONLL_03()
wnut17: Corpus = WNUT_17()

# ----- TAG SPACES -----
conll_dict = conll03.make_label_dictionary(task)
wnut_dict = wnut17.make_label_dictionary(task)

# ----- SHARED EMBEDDING LAYERS -----
shared_embedding_Layer = TransformerWordEmbeddings(model="bert-base-uncased")

# ----- TASKS -----
conll_tagger: SequenceTagger = SequenceTagger(embeddings=shared_embedding_Layer,
                                            tag_dictionary=conll_dict,
                                            tag_type=task,
                                            use_rnn=True,
                                            use_crf=True,
                                            hidden_size=256)

wnut_tagger: SequenceTagger = SequenceTagger(embeddings=shared_embedding_Layer,
                                            tag_dictionary=wnut_dict,
                                            tag_type=task,
                                            use_rnn=True,
                                            use_crf=True,
                                            hidden_size=256)


# ----- MULTITASK CORPUS -----
multi_corpus = MultitaskCorpus(
    {"corpus": conll03, "model": conll_tagger},
    {"corpus": wnut17, "model": wnut_tagger}
)

# ----- MULTITASK MODEL -----
multitask_model: MultitaskModel = MultitaskModel(multi_corpus.models)

# ----- TRAINING ON MODEL AND CORPUS -----
trainer: ModelTrainer = ModelTrainer(multitask_model, multi_corpus)
trainer.train(f'resources/taggers/multitask-example/',
                learning_rate=0.01,
                mini_batch_size=32,
                max_epochs=3,
                monitor_test=False,
                embeddings_storage_mode="none",
                weight_decay=0.)

# The first task from multitask corpus will be task_0, second one task_1, and so on.
res_task0 = multitask_model.task_0.evaluate(conll03.test, task)
print(res_task0)

res_task1 = multitask_model.task_1.evaluate(wnut17.test, task)
print(res_task1)
```

## Example 2: Combine sequence labeling and text classification
```python
import flair
from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CONLL_03, TREC_6
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import MultitaskModel, SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from transformers import AutoTokenizer, AutoModel, AutoConfig

flair.device = "cuda:0"

# ----- CORPORA -----
conll03: Corpus = CONLL_03()
trec6: Corpus = TREC_6()


# ----- TAG SPACES -----
conll_dictionary = conll03.make_label_dictionary('ner')
trec6_dictionary = trec6.make_label_dictionary('question_class')

# ----- TOKENIZER + MODEL -----
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
model = AutoModel.from_pretrained(model_name, config=config)
lm = {"model": model, "tokenizer": tokenizer}

# ----- SHARED EMBEDDING LAYERs -----
shared_embedding_Layer = TransformerWordEmbeddings(model=lm)
shared_document_embedding_layer = TransformerDocumentEmbeddings(model=lm)

# ----- TASKS -----
conll_tagger: SequenceTagger = SequenceTagger(embeddings=shared_embedding_Layer,
                                            tag_dictionary=conll_dictionary,
                                            tag_type='ner',
                                            use_rnn=True,
                                            use_crf=True,
                                            hidden_size=256)

trec_classifier = TextClassifier(document_embeddings=shared_document_embedding_layer,
                                 label_dictionary=trec6_dictionary,
                                 label_type='question_class')

# ----- MULTITASK CORPUS -----
multi_corpus = MultitaskCorpus(
    {"corpus": conll03, "model": conll_tagger},
    {"corpus": trec6, "model": trec_classifier}
)

# ----- MULTITASK MODEL -----
multitask_model: MultitaskModel = MultitaskModel(multi_corpus.models)

# ----- TRAINING ON MODEL AND CORPUS -----
trainer: ModelTrainer = ModelTrainer(multitask_model, multi_corpus)
trainer.train(f'resources/taggers/multi-corpus/',
                learning_rate=0.01,
                mini_batch_size=32,
                max_epochs=1,
                monitor_test=False,
                embeddings_storage_mode="none",
                weight_decay=0.)

res_task0 = multitask_model.task_0.evaluate(conll03.test, "ner")
print(res_task0)
res_task1 = multitask_model.task_1.evaluate(trec6.test, 'question_class')
print(res_task1)
```

You can also swap out `TransformerWordEmbeddings` or `TransformerDocumentEmbeddings`
with all other embedding types such as `DocumentRNNEmbeddings` or `CharacterEmbeddings`.