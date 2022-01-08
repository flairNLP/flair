from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def get_stackoverflow_labels():
    with open("evaluation/StackOverflow/label_StackOverflow.txt", "r", encoding="utf8") as myfile:
        data = myfile.readlines()
        return data


def get_stackoverflow_data():
    with open("evaluation/StackOverflow/title_StackOverflow.txt", "r", encoding="utf8") as myfile:
        data = myfile.readlines()
        return data


maxDocuments = 400
categories = ["rec.motorcycles", "rec.sport.baseball", "comp.graphics", "sci.space", "talk.politics.mideast"]


def get_20news_data_smaller():
    ng5 = fetch_20newsgroups(categories=categories)
    return ng5.data[1:maxDocuments]


def get_20_news_label_smaller():
    ng5 = fetch_20newsgroups(categories=categories)
    return ng5.target[1:maxDocuments]


def get_20news_data():
    ng5 = fetch_20newsgroups()
    return ng5.data


def get_20_news_label():
    ng5 = fetch_20newsgroups()
    return ng5.target


def evaluate(labels: list, predict_labels: list):
    acc = accuracy_score(labels, predict_labels)
    nmi = normalized_mutual_info_score(labels, predict_labels)
    print("ACC: " + str(acc))
    print("NMI: " + str(nmi))
