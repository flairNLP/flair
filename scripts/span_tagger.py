from flair.data import Sentence
from flair.models import MultitaskModel

# For comparison: This works since the label type is "ner" for both models in the multitask model
classifier: MultitaskModel = MultitaskModel.load("zelda")

sentence = Sentence("Kirk and Spock met on the Enterprise")

classifier.predict(sentence)

print(sentence)

# Giving them sensible label names, now made possible with this PR
classifier.tasks["Task_1"]._label_type = "nel"
classifier.tasks["Task_1"]._span_label_type = "ner"

# However, this no longer makes predictions
sentence = Sentence("Kirk and Spock met on the Enterprise")

classifier.predict(sentence)

print(sentence)
