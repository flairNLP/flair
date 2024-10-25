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
    labels: list[Label] = sentence.get_labels("pos")
    assert len(labels) == 3
    assert labels[0].data_point.text == "I"
    assert labels[0].value == "pronoun"
    assert labels[1].data_point.text == "love"
    assert labels[1].value == "verb"
    assert labels[2].data_point.text == "Berlin"
    assert labels[2].value == "proper noun"

    # check if there are is one SENTIMENT label with correct text and values
    labels: list[Label] = sentence.get_labels("sentiment")
    assert len(labels) == 1
    assert labels[0].data_point.text == "love"
    assert labels[0].value == "positive"

    # check if all tokens are correctly labeled
    assert len(sentence) == 3
    assert sentence[0].text == "I"
    assert sentence[1].text == "love"
    assert sentence[2].text == "Berlin"
    assert len(sentence[0].get_labels("pos")) == 1
    assert len(sentence[1].get_labels("pos")) == 1
    assert len(sentence[1].labels) == 2
    assert len(sentence[2].get_labels("pos")) == 1

    assert sentence[1].get_label("pos").value == "verb"
    assert sentence[1].get_label("sentiment").value == "positive"

    # remove the pos label from the last word
    sentence[2].remove_labels("pos")
    # there should be 2 POS labels left
    labels: list[Label] = sentence.get_labels("pos")
    assert len(labels) == 2
    assert len(sentence[0].get_labels("pos")) == 1
    assert len(sentence[1].get_labels("pos")) == 1
    assert len(sentence[1].labels) == 2
    assert len(sentence[2].get_labels("pos")) == 0

    # now remove all pos tags
    sentence.remove_labels("pos")
    print(sentence[0].get_labels("pos"))
    assert len(sentence.get_labels("pos")) == 0
    assert len(sentence.get_labels("sentiment")) == 1
    assert len(sentence.labels) == 1

    assert len(sentence[0].get_labels("pos")) == 0
    assert len(sentence[1].get_labels("pos")) == 0
    assert len(sentence[2].get_labels("pos")) == 0


def test_span_tags():
    # set 3 labels for 2 spans (HU is tagged twice)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")
    sentence[0:4].add_label("ner", "Organization")
    sentence[0:4].add_label("ner", "University")
    sentence[7:8].add_label("ner", "City")

    # check if there are three labels with correct text and values
    labels: list[Label] = sentence.get_labels("ner")
    assert len(labels) == 3
    assert labels[0].data_point.text == "Humboldt Universität zu Berlin"
    assert labels[0].value == "Organization"
    assert labels[1].data_point.text == "Humboldt Universität zu Berlin"
    assert labels[1].value == "University"
    assert labels[2].data_point.text == "Berlin"
    assert labels[2].value == "City"

    # check if there are two spans with correct text and values
    spans: list[Span] = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "Humboldt Universität zu Berlin"
    assert len(spans[0].get_labels("ner")) == 2
    assert spans[1].text == "Berlin"
    assert spans[1].get_label("ner").value == "City"

    # now delete the NER tags of "Humboldt-Universität zu Berlin"
    sentence[0:4].remove_labels("ner")
    # should be only one NER label left
    labels: list[Label] = sentence.get_labels("ner")
    assert len(labels) == 1
    assert labels[0].data_point.text == "Berlin"
    assert labels[0].value == "City"
    # and only one NER span
    spans: list[Span] = sentence.get_spans("ner")
    assert len(spans) == 1
    assert spans[0].text == "Berlin"
    assert spans[0].get_label("ner").value == "City"


def test_different_span_tags():
    # set 3 labels for 2 spans (HU is tagged twice with different tags)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")
    sentence[0:4].add_label("ner", "Organization")
    sentence[0:4].add_label("orgtype", "University")
    sentence[7:8].add_label("ner", "City")

    # check if there are three labels with correct text and values
    labels: list[Label] = sentence.get_labels("ner")
    assert len(labels) == 2
    assert labels[0].data_point.text == "Humboldt Universität zu Berlin"
    assert labels[0].value == "Organization"
    assert labels[1].data_point.text == "Berlin"
    assert labels[1].value == "City"

    # check if there are two spans with correct text and values
    spans: list[Span] = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "Humboldt Universität zu Berlin"
    assert spans[0].get_label("ner").value == "Organization"
    assert spans[0].get_label("orgtype").value == "University"
    assert len(spans[0].get_labels("ner")) == 1
    assert spans[1].text == "Berlin"
    assert spans[1].get_label("ner").value == "City"

    # now delete the NER tags of "Humboldt-Universität zu Berlin"
    sentence[0:4].remove_labels("ner")
    # should be only one NER label left
    labels: list[Label] = sentence.get_labels("ner")
    assert len(labels) == 1
    assert labels[0].data_point.text == "Berlin"
    assert labels[0].value == "City"
    # and only one NER span
    spans: list[Span] = sentence.get_spans("ner")
    assert len(spans) == 1
    assert spans[0].text == "Berlin"
    assert spans[0].get_label("ner").value == "City"
    # but there is also one orgtype span and label
    labels: list[Label] = sentence.get_labels("orgtype")
    assert len(labels) == 1
    assert labels[0].data_point.text == "Humboldt Universität zu Berlin"
    assert labels[0].value == "University"
    # and only one NER span
    spans: list[Span] = sentence.get_spans("orgtype")
    assert len(spans) == 1
    assert spans[0].text == "Humboldt Universität zu Berlin"
    assert spans[0].get_label("orgtype").value == "University"

    # let's add the NER tag back
    sentence[0:4].add_label("ner", "Organization")
    # check if there are three labels with correct text and values
    labels: list[Label] = sentence.get_labels("ner")
    print(labels)
    assert len(labels) == 2
    assert labels[0].data_point.text == "Humboldt Universität zu Berlin"
    assert labels[0].value == "Organization"
    assert labels[1].data_point.text == "Berlin"
    assert labels[1].value == "City"

    # check if there are two spans with correct text and values
    spans: list[Span] = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "Humboldt Universität zu Berlin"
    assert spans[0].get_label("ner").value == "Organization"
    assert spans[0].get_label("orgtype").value == "University"
    assert len(spans[0].get_labels("ner")) == 1
    assert spans[1].text == "Berlin"
    assert spans[1].get_label("ner").value == "City"

    # now remove all NER tags
    sentence.remove_labels("ner")
    assert len(sentence.get_labels("ner")) == 0
    assert len(sentence.get_spans("ner")) == 0
    assert len(sentence.get_spans("orgtype")) == 1
    assert len(sentence.get_labels("orgtype")) == 1
    assert len(sentence.labels) == 1

    assert len(sentence[0:4].get_labels("ner")) == 0
    assert len(sentence[0:4].get_labels("orgtype")) == 1


def test_relation_tags():
    # set 3 labels for 2 spans (HU is tagged twice with different tags)
    sentence = Sentence("Humboldt Universität zu Berlin is located in Berlin .")

    # create two relation label
    Relation(sentence[0:4], sentence[7:8]).add_label("rel", "located in")
    Relation(sentence[0:2], sentence[3:4]).add_label("rel", "university of")
    Relation(sentence[0:2], sentence[3:4]).add_label("syntactic", "apposition")

    # there should be two relation labels
    labels: list[Label] = sentence.get_labels("rel")
    assert len(labels) == 2
    assert labels[0].value == "located in"
    assert labels[1].value == "university of"

    # there should be one syntactic labels
    labels: list[Label] = sentence.get_labels("syntactic")
    assert len(labels) == 1

    # there should be two relations, one with two and one with one label
    relations: list[Relation] = sentence.get_relations("rel")
    assert len(relations) == 2
    assert len(relations[0].labels) == 1
    assert len(relations[1].labels) == 2


def test_sentence_labels():
    # example sentence
    sentence = Sentence("I love Berlin")
    sentence.add_label("sentiment", "positive")
    sentence.add_label("topic", "travelling")

    assert len(sentence.labels) == 2
    assert len(sentence.get_labels("sentiment")) == 1
    assert len(sentence.get_labels("topic")) == 1

    # add another topic label
    sentence.add_label("topic", "travelling")
    assert len(sentence.labels) == 3
    assert len(sentence.get_labels("sentiment")) == 1
    assert len(sentence.get_labels("topic")) == 2

    sentence.remove_labels("topic")
    assert len(sentence.labels) == 1
    assert len(sentence.get_labels("sentiment")) == 1
    assert len(sentence.get_labels("topic")) == 0


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
    assert len(sentence.labels) == 6
    assert len(sentence.get_labels("pos")) == 4
    assert len(sentence.get_labels("sentiment")) == 1
    assert len(sentence.get_labels("ner")) == 1


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
