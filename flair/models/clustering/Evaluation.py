from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def getStackOverFlowLabels():
    with open("evaluation/StackOverflow/title_StackOverflow.txt", "r", encoding="utf8") as myfile:
        data = myfile.readlines()
        return data


def getStackOverFlowData():
    with open("evaluation/StackOverflow/title_StackOverflow.txt", "r", encoding="utf8") as myfile:
        data = myfile.readlines()
        return data


maxDocuments = 400
categories = [
    'rec.motorcycles',
    'rec.sport.baseball',
    'comp.graphics',
    'sci.space',
    'talk.politics.mideast'
]


def get20NewsData():
    ng5 = fetch_20newsgroups(categories=categories)
    return ng5.data[1:maxDocuments]


def get20NewsLabel():
    ng5 = fetch_20newsgroups(categories=categories)
    return ng5.target[1:maxDocuments]


def evaluate(labels: list, predict_labels: list):
    acc = accuracy_score(labels, predict_labels)
    nmi = normalized_mutual_info_score(labels, predict_labels)
    print("ACC: " + str(acc))
    print("NMI: " + str(nmi))
