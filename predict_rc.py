from flair.data import Sentence
from flair.models import RelationClassifier

classifier: RelationClassifier = RelationClassifier.load("./resources/classifiers/example-rc/best-model.pt")

# sentence = Sentence("The most common audits were about waste and recycling .".split(" "))
# for token, tag in zip(sentence.tokens, ["O", "O", "O", "B-E1", "O", "O", "B-E2", "O", "O", "O"]):
#     token.set_label("ner", tag)

sentence = Sentence("The company fabricates plastic chairs .".split(" "))
for token, tag in zip(sentence.tokens, ["O", "B-E1", "O", "O", "B-E2", "O"]):
    token.set_label("ner", tag)

classifier.predict(sentence)

print("Analysing %s" % sentence)
print("\nThe following relations are found: \n")
print(sentence.relations)
