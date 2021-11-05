# Tutorial 11: Multitask Learning

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
how multitask models are built in flair: bottom-up. This means, you 
have to create shared layers for your tasks first before you finally
create a model.

In this context, a task is a combination of a corpus and a model, indicating
what task should be performed through which model. In the end, you will
ensemble one flair model per corpus even though if you share layers between
different tasks.

```python
from flair.models import TARSClassifier
from flair.data import Sentence
var = x
```

The output should look like:

```console
console
```