from typing import List

from flair.data import Label, Relation, Sentence, Span


def test_token_tags():
    # example sentence
    sentence = Sentence("I love Berlin")

    # set 4 labels for 2 tokens ('love' is tagged twice)
    sentence[1].add_label("pos", "verb")
    sentence[1].add_label("sentiment", "positive")
    sentence[2].add_label("pos", "proper noun")
    sentence[0].add_label("pos", "pronoun")

    # check if there are three POS labels with correct text and values
    labels: List[Label] = sentence.get_labels("pos")
    assert 3 == len(labels)
    assert "I" == labels[0].data_point.text
    assert "pronoun" == labels[0].value
    assert "love" == labels[1].data_point.text
    assert "verb" == labels[1].value
    assert "Berlin" == labels[2].data_point.text
    assert "proper noun" == labels[2].value

    # check if there are is one SENTIMENT label with correct text and values
    labels: List[Label] = sentence.get_labels("sentiment")
    assert 1 == len(labels)
    assert "love" == labels[0].data_point.text
    assert "positive" == labels[0].value

    # check if all tokens are correctly labeled
    assert 3 == len(sentence)
    assert "I" == sentence[0].text
    assert "love" == sentence[1].text
    assert "Berlin" == sentence[2].text
    assert 1 == len(sentence[0].get_labels("pos"))
    assert 1 == len(sentence[1].get_labels("pos"))
    assert 2 == len(sentence[1].labels)
    assert 1 == len(sentence[2].get_labels("pos"))

    assert "verb" == sentence[1].get_label("pos").value
    assert "positive" == sentence[1].get_label("sentiment").value

    # remove the pos label from the last word
    sentence[2].remove_labels("pos")
    # there should be 2 POS labels left
    labels: List[Label] = sentence.get_labels("pos")
    assert 2 == len(labels)
    assert 1 == len(sentence[0].get_labels("pos"))
    assert 1 == len(sentence[1].get_labels("pos"))
    assert 2 == len(sentence[1].labels)
    assert 0 == len(sentence[2].get_labels("pos"))

    # now remove all pos tags
    sentence.remove_labels("pos")
    print(sentence[0].get_labels("pos"))
    assert 0 == len(sentence.get_labels("pos"))
    assert 1 == len(sentence.get_labels("sentiment"))
    assert 1 == len(sentence.labels)

    assert 0 == len(sentence[0].get_labels("pos"))
    assert 0 == len(sentence[1].get_labels("pos"))
    assert 0 == len(sentence[2].get_labels("pos"))


def test_span_tags():

    # set 3 labels for 2 spans (HU is tagged twice)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")
    sentence[0:4].add_label("ner", "Organization")
    sentence[0:4].add_label("ner", "University")
    sentence[7:8].add_label("ner", "City")

    # check if there are three labels with correct text and values
    labels: List[Label] = sentence.get_labels("ner")
    assert 3 == len(labels)
    assert "Humboldt Universität zu Berlin" == labels[0].data_point.text
    assert "Organization" == labels[0].value
    assert "Humboldt Universität zu Berlin" == labels[1].data_point.text
    assert "University" == labels[1].value
    assert "Berlin" == labels[2].data_point.text
    assert "City" == labels[2].value

    # check if there are two spans with correct text and values
    spans: List[Span] = sentence.get_spans("ner")
    assert 2 == len(spans)
    assert "Humboldt Universität zu Berlin" == spans[0].text
    assert 2 == len(spans[0].get_labels("ner"))
    assert "Berlin" == spans[1].text
    assert "City" == spans[1].get_label("ner").value

    # now delete the NER tags of "Humboldt-Universität zu Berlin"
    sentence[0:4].remove_labels("ner")
    # should be only one NER label left
    labels: List[Label] = sentence.get_labels("ner")
    assert 1 == len(labels)
    assert "Berlin" == labels[0].data_point.text
    assert "City" == labels[0].value
    # and only one NER span
    spans: List[Span] = sentence.get_spans("ner")
    assert 1 == len(spans)
    assert "Berlin" == spans[0].text
    assert "City" == spans[0].get_label("ner").value


def test_different_span_tags():

    # set 3 labels for 2 spans (HU is tagged twice with different tags)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")
    sentence[0:4].add_label("ner", "Organization")
    sentence[0:4].add_label("orgtype", "University")
    sentence[7:8].add_label("ner", "City")

    # check if there are three labels with correct text and values
    labels: List[Label] = sentence.get_labels("ner")
    assert 2 == len(labels)
    assert "Humboldt Universität zu Berlin" == labels[0].data_point.text
    assert "Organization" == labels[0].value
    assert "Berlin" == labels[1].data_point.text
    assert "City" == labels[1].value

    # check if there are two spans with correct text and values
    spans: List[Span] = sentence.get_spans("ner")
    assert 2 == len(spans)
    assert "Humboldt Universität zu Berlin" == spans[0].text
    assert "Organization" == spans[0].get_label("ner").value
    assert "University" == spans[0].get_label("orgtype").value
    assert 1 == len(spans[0].get_labels("ner"))
    assert "Berlin" == spans[1].text
    assert "City" == spans[1].get_label("ner").value

    # now delete the NER tags of "Humboldt-Universität zu Berlin"
    sentence[0:4].remove_labels("ner")
    # should be only one NER label left
    labels: List[Label] = sentence.get_labels("ner")
    assert 1 == len(labels)
    assert "Berlin" == labels[0].data_point.text
    assert "City" == labels[0].value
    # and only one NER span
    spans: List[Span] = sentence.get_spans("ner")
    assert 1 == len(spans)
    assert "Berlin" == spans[0].text
    assert "City" == spans[0].get_label("ner").value
    # but there is also one orgtype span and label
    labels: List[Label] = sentence.get_labels("orgtype")
    assert 1 == len(labels)
    assert "Humboldt Universität zu Berlin" == labels[0].data_point.text
    assert "University" == labels[0].value
    # and only one NER span
    spans: List[Span] = sentence.get_spans("orgtype")
    assert 1 == len(spans)
    assert "Humboldt Universität zu Berlin" == spans[0].text
    assert "University" == spans[0].get_label("orgtype").value

    # let's add the NER tag back
    sentence[0:4].add_label("ner", "Organization")
    # check if there are three labels with correct text and values
    labels: List[Label] = sentence.get_labels("ner")
    print(labels)
    assert 2 == len(labels)
    assert "Humboldt Universität zu Berlin" == labels[0].data_point.text
    assert "Organization" == labels[0].value
    assert "Berlin" == labels[1].data_point.text
    assert "City" == labels[1].value

    # check if there are two spans with correct text and values
    spans: List[Span] = sentence.get_spans("ner")
    assert 2 == len(spans)
    assert "Humboldt Universität zu Berlin" == spans[0].text
    assert "Organization" == spans[0].get_label("ner").value
    assert "University" == spans[0].get_label("orgtype").value
    assert 1 == len(spans[0].get_labels("ner"))
    assert "Berlin" == spans[1].text
    assert "City" == spans[1].get_label("ner").value

    # now remove all NER tags
    sentence.remove_labels("ner")
    assert 0 == len(sentence.get_labels("ner"))
    assert 0 == len(sentence.get_spans("ner"))
    assert 1 == len(sentence.get_spans("orgtype"))
    assert 1 == len(sentence.get_labels("orgtype"))
    assert 1 == len(sentence.labels)

    assert 0 == len(sentence[0:4].get_labels("ner"))
    assert 1 == len(sentence[0:4].get_labels("orgtype"))


def test_relation_tags():
    # set 3 labels for 2 spans (HU is tagged twice with different tags)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")

    # create two relation label
    Relation(sentence[0:4], sentence[7:8]).add_label("rel", "located in")
    Relation(sentence[0:2], sentence[3:4]).add_label("rel", "university of")
    Relation(sentence[0:2], sentence[3:4]).add_label("syntactic", "apposition")

    # there should be two relation labels
    labels: List[Label] = sentence.get_labels("rel")
    assert 2 == len(labels)
    assert "located in" == labels[0].value
    assert "university of" == labels[1].value

    # there should be one syntactic labels
    labels: List[Label] = sentence.get_labels("syntactic")
    assert 1 == len(labels)

    # there should be two relations, one with two and one with one label
    relations: List[Relation] = sentence.get_relations("rel")
    assert 2 == len(relations)
    assert 1 == len(relations[0].labels)
    assert 2 == len(relations[1].labels)


def test_sentence_labels():
    # example sentence
    sentence = Sentence("I love Berlin")
    sentence.add_label("sentiment", "positive")
    sentence.add_label("topic", "travelling")

    assert 2 == len(sentence.labels)
    assert 1 == len(sentence.get_labels("sentiment"))
    assert 1 == len(sentence.get_labels("topic"))

    # add another topic label
    sentence.add_label("topic", "travelling")
    assert 3 == len(sentence.labels)
    assert 1 == len(sentence.get_labels("sentiment"))
    assert 2 == len(sentence.get_labels("topic"))

    sentence.remove_labels("topic")
    assert 1 == len(sentence.labels)
    assert 1 == len(sentence.get_labels("sentiment"))
    assert 0 == len(sentence.get_labels("topic"))


def test_mixed_labels():
    # example sentence
    sentence = Sentence("I love New York")

    # has sentiment value
    sentence.add_label("sentiment", "positive")

    # has 4 part of speech tags
    sentence[1].add_label("pos", "verb")
    sentence[2].add_label("pos", "proper noun")
    sentence[3].add_label("pos", "proper noun")
    sentence[0].add_label("pos", "pronoun")

    # has 1 NER tag
    sentence[2:4].add_label("ner", "City")

    # should be in total 6 labels
    assert 6 == len(sentence.labels)
    assert 4 == len(sentence.get_labels("pos"))
    assert 1 == len(sentence.get_labels("sentiment"))
    assert 1 == len(sentence.get_labels("ner"))


def test_data_point_equality():

    # example sentence
    sentence = Sentence("George Washington went to Washington .")

    # add two NER labels
    sentence[0:2].add_label("span_ner", "PER")
    sentence[0:2].add_label("span_other", "Politician")
    sentence[4].add_label("ner", "LOC")
    sentence[4].add_label("other", "Village")

    # get the four labels
    ner_label = sentence.get_label("ner")
    other_label = sentence.get_label("other")
    span_ner_label = sentence.get_label("span_ner")
    span_other_label = sentence.get_label("span_other")

    # check that only two of the respective data points are equal
    assert ner_label.data_point == other_label.data_point
    assert span_ner_label.data_point == span_other_label.data_point
    assert ner_label.data_point != span_other_label.data_point
    assert other_label.data_point != span_ner_label.data_point
