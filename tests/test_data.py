import os
from typing import List

import pytest

import flair.datasets
from flair.data import (
    Sentence,
    Label,
    Token,
    Dictionary,
    Corpus,
    Span,
    segtok_tokenizer,
    build_japanese_tokenizer
)


def test_get_head():
    token1 = Token("I", 0)
    token2 = Token("love", 1, 0)
    token3 = Token("Berlin", 2, 1)

    sentence: Sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)

    assert token2 == token3.get_head()
    assert token1 == token2.get_head()
    assert None == token1.get_head()


def test_create_sentence_on_empty_string():

    sentence: Sentence = Sentence("")
    assert 0 == len(sentence.tokens)


def test_create_sentence_without_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.")

    assert 3 == len(sentence.tokens)
    assert "I" == sentence.tokens[0].text
    assert "love" == sentence.tokens[1].text
    assert "Berlin." == sentence.tokens[2].text


# skip because it is optional https://github.com/flairNLP/flair/pull/1296
# def test_create_sentence_using_japanese_tokenizer():
#     sentence: Sentence = Sentence("私はベルリンが好き", use_tokenizer=build_japanese_tokenizer())
#
#     assert 5 == len(sentence.tokens)
#     assert "私" == sentence.tokens[0].text
#     assert "は" == sentence.tokens[1].text
#     assert "ベルリン" == sentence.tokens[2].text
#     assert "が" == sentence.tokens[3].text
#     assert "好き" == sentence.tokens[4].text


def test_token_indices():

    text = ":    nation on"
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = ":    nation on"
    sentence = Sentence(text, use_tokenizer=segtok_tokenizer)
    assert text == sentence.to_original_text()

    text = "I love Berlin."
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = 'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = 'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text, use_tokenizer=segtok_tokenizer)
    assert text == sentence.to_original_text()


def test_create_sentence_with_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=segtok_tokenizer)

    assert 4 == len(sentence.tokens)
    assert "I" == sentence.tokens[0].text
    assert "love" == sentence.tokens[1].text
    assert "Berlin" == sentence.tokens[2].text
    assert "." == sentence.tokens[3].text


def test_sentence_to_plain_string():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=segtok_tokenizer)

    assert "I love Berlin ." == sentence.to_tokenized_string()


def test_sentence_to_real_string(tasks_base_path):
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=segtok_tokenizer)
    assert "I love Berlin." == sentence.to_plain_string()

    corpus = flair.datasets.GERMEVAL_14(base_path=tasks_base_path)

    sentence = corpus.train[0]
    assert (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
        == sentence.to_tokenized_string()
    )
    assert (
        'Schartau sagte dem "Tagesspiegel" vom Freitag, Fischer sei "in einer Weise aufgetreten, die alles andere als überzeugend war".'
        == sentence.to_plain_string()
    )

    sentence = corpus.train[1]
    assert (
        "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als Möbelvertreter , als er einen fliegenden Händler aus dem Libanon traf ."
        == sentence.to_tokenized_string()
    )
    assert (
        "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als Möbelvertreter, als er einen fliegenden Händler aus dem Libanon traf."
        == sentence.to_plain_string()
    )


def test_sentence_infer_tokenization():
    sentence: Sentence = Sentence()
    sentence.add_token(Token("xyz"))
    sentence.add_token(Token('"'))
    sentence.add_token(Token("abc"))
    sentence.add_token(Token('"'))
    sentence.infer_space_after()

    assert 'xyz " abc "' == sentence.to_tokenized_string()
    assert 'xyz "abc"' == sentence.to_plain_string()

    sentence: Sentence = Sentence('xyz " abc "')
    sentence.infer_space_after()
    assert 'xyz " abc "' == sentence.to_tokenized_string()
    assert 'xyz "abc"' == sentence.to_plain_string()


def test_sentence_get_item():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=segtok_tokenizer)

    assert sentence.get_token(1) == sentence[0]
    assert sentence.get_token(3) == sentence[2]

    with pytest.raises(IndexError):
        token = sentence[4]


def test_sentence_whitespace_tokenization():
    sentence: Sentence = Sentence("I  love Berlin .")

    assert 4 == len(sentence.tokens)
    assert "I" == sentence.get_token(1).text
    assert "love" == sentence.get_token(2).text
    assert "Berlin" == sentence.get_token(3).text
    assert "." == sentence.get_token(4).text


def test_sentence_to_tagged_string():
    token1 = Token("I", 0)
    token2 = Token("love", 1, 0)
    token3 = Token("Berlin", 2, 1)
    token3.add_tag("ner", "LOC")

    sentence: Sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)

    assert "I love Berlin <LOC>" == sentence.to_tagged_string()


def test_sentence_add_token():
    token1: Token = Token("Munich")
    token2: Token = Token("and")
    token3: Token = Token("Berlin")
    token4: Token = Token("are")
    token5: Token = Token("nice")

    sentence: Sentence = Sentence()

    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)
    sentence.add_token(token4)
    sentence.add_token(token5)

    sentence.add_token("cities")
    sentence.add_token(Token("."))

    assert "Munich and Berlin are nice cities ." == sentence.to_tokenized_string()


def test_dictionary_get_items_with_unk():
    dictionary: Dictionary = Dictionary()

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert 4 == len(items)
    assert "<unk>" == items[0]
    assert "class_1" == items[1]
    assert "class_2" == items[2]
    assert "class_3" == items[3]


def test_dictionary_get_items_without_unk():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    items = dictionary.get_items()

    assert 3 == len(items)
    assert "class_1" == items[0]
    assert "class_2" == items[1]
    assert "class_3" == items[2]


def test_dictionary_get_idx_for_item():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    idx = dictionary.get_idx_for_item("class_2")

    assert 1 == idx


def test_dictionary_get_item_for_index():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    item = dictionary.get_item_for_index(0)

    assert "class_1" == item


def test_dictionary_save_and_load():
    dictionary: Dictionary = Dictionary(add_unk=False)

    dictionary.add_item("class_1")
    dictionary.add_item("class_2")
    dictionary.add_item("class_3")

    file_path = "dictionary.txt"

    dictionary.save(file_path)
    loaded_dictionary = dictionary.load_from_file(file_path)

    assert len(dictionary) == len(loaded_dictionary)
    assert len(dictionary.get_items()) == len(loaded_dictionary.get_items())

    # clean up file
    os.remove(file_path)


def test_tagged_corpus_get_all_sentences():
    train_sentence = Sentence("I'm used in training.", use_tokenizer=segtok_tokenizer)
    dev_sentence = Sentence("I'm a dev sentence.", use_tokenizer=segtok_tokenizer)
    test_sentence = Sentence(
        "I will be only used for testing.", use_tokenizer=segtok_tokenizer
    )

    corpus: Corpus = Corpus([train_sentence], [dev_sentence], [test_sentence])

    all_sentences = corpus.get_all_sentences()

    assert 3 == len(all_sentences)


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence(
        "used in training. training is cool.", use_tokenizer=segtok_tokenizer
    )

    corpus: Corpus = Corpus([train_sentence], [], [])

    vocab = corpus.make_vocab_dictionary(max_tokens=2, min_freq=-1)

    assert 3 == len(vocab)
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=-1)

    assert 7 == len(vocab)

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=2)

    assert 3 == len(vocab)
    assert "<unk>" in vocab.get_items()
    assert "training" in vocab.get_items()
    assert "." in vocab.get_items()


def test_label_set_confidence():
    label = Label("class_1", 3.2)

    assert 1.0 == label.score
    assert "class_1" == label.value

    label.score = 0.2

    assert 0.2 == label.score


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence("sentence 1").add_label('label', 'class_1')

    sentence_2 = Sentence("sentence 2").add_label('label', 'class_2')

    sentence_3 = Sentence("sentence 3").add_label('label', 'class_1')

    corpus: Corpus = Corpus([sentence_1, sentence_2, sentence_3], [], [])

    label_dict = corpus.make_label_dictionary('label')

    assert 2 == len(label_dict)
    assert "<unk>" not in label_dict.get_items()
    assert "class_1" in label_dict.get_items()
    assert "class_2" in label_dict.get_items()


def test_tagged_corpus_statistics():

    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label('label', 'class_1')

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label('label', 'class_2')

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True).add_label('label', 'class_1')

    class_to_count_dict = Corpus._count_sentence_labels(
        [train_sentence, dev_sentence, test_sentence]
    )

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert 2 == class_to_count_dict["class_1"]
    assert 1 == class_to_count_dict["class_2"]

    tokens_in_sentences = Corpus._get_tokens_per_sentence(
        [train_sentence, dev_sentence, test_sentence]
    )

    assert 3 == len(tokens_in_sentences)
    assert 4 == tokens_in_sentences[0]
    assert 5 == tokens_in_sentences[1]
    assert 4 == tokens_in_sentences[2]


def test_tagged_corpus_statistics_multi_label():

    train_sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label('label', 'class_1')

    dev_sentence = Sentence("The sun is shining.", use_tokenizer=True).add_label('label', 'class_2')

    test_sentence = Sentence("Berlin is sunny.", use_tokenizer=True)
    test_sentence.add_label('label', 'class_1')
    test_sentence.add_label('label', 'class_2')

    class_to_count_dict = Corpus._count_sentence_labels(
        [train_sentence, dev_sentence, test_sentence]
    )

    assert "class_1" in class_to_count_dict
    assert "class_2" in class_to_count_dict
    assert 2 == class_to_count_dict["class_1"]
    assert 2 == class_to_count_dict["class_2"]

    tokens_in_sentences = Corpus._get_tokens_per_sentence(
        [train_sentence, dev_sentence, test_sentence]
    )

    assert 3 == len(tokens_in_sentences)
    assert 4 == tokens_in_sentences[0]
    assert 5 == tokens_in_sentences[1]
    assert 4 == tokens_in_sentences[2]


def test_tagged_corpus_get_tag_statistic():
    train_sentence = Sentence("Zalando Research is located in Berlin .")
    train_sentence[0].add_tag("ner", "B-ORG")
    train_sentence[1].add_tag("ner", "E-ORG")
    train_sentence[5].add_tag("ner", "S-LOC")

    dev_sentence = Sentence(
        "Facebook, Inc. is a company, and Google is one as well.",
        use_tokenizer=segtok_tokenizer,
    )
    dev_sentence[0].add_tag("ner", "B-ORG")
    dev_sentence[1].add_tag("ner", "I-ORG")
    dev_sentence[2].add_tag("ner", "E-ORG")
    dev_sentence[8].add_tag("ner", "S-ORG")

    test_sentence = Sentence("Nothing to do with companies.")

    tag_to_count_dict = Corpus._count_token_labels(
        [train_sentence, dev_sentence, test_sentence], "ner"
    )

    assert 1 == tag_to_count_dict["S-ORG"]
    assert 1 == tag_to_count_dict["S-LOC"]
    assert 2 == tag_to_count_dict["B-ORG"]
    assert 2 == tag_to_count_dict["E-ORG"]
    assert 1 == tag_to_count_dict["I-ORG"]


def test_tagged_corpus_downsample():

    sentence = Sentence("I love Berlin.", use_tokenizer=True).add_label('label', 'class_1')

    corpus: Corpus = Corpus(
        [
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
            sentence,
        ],
        [],
        [],
    )

    assert 10 == len(corpus.train)

    corpus.downsample(percentage=0.3, downsample_dev=False, downsample_test=False)

    assert 3 == len(corpus.train)


def test_spans():
    sentence = Sentence("Zalando Research is located in Berlin .")

    # bioes tags
    sentence[0].add_tag("ner", "B-ORG")
    sentence[1].add_tag("ner", "E-ORG")
    sentence[5].add_tag("ner", "S-LOC")

    spans: List[Span] = sentence.get_spans("ner")

    assert 2 == len(spans)
    assert "Zalando Research" == spans[0].text
    assert "ORG" == spans[0].tag
    assert "Berlin" == spans[1].text
    assert "LOC" == spans[1].tag

    # bio tags
    sentence[0].add_tag("ner", "B-ORG")
    sentence[1].add_tag("ner", "I-ORG")
    sentence[5].add_tag("ner", "B-LOC")

    spans: List[Span] = sentence.get_spans("ner")

    assert "Zalando Research" == spans[0].text
    assert "ORG" == spans[0].tag
    assert "Berlin" == spans[1].text
    assert "LOC" == spans[1].tag

    # broken tags
    sentence[0].add_tag("ner", "I-ORG")
    sentence[1].add_tag("ner", "E-ORG")
    sentence[5].add_tag("ner", "I-LOC")

    spans: List[Span] = sentence.get_spans("ner")

    assert "Zalando Research" == spans[0].text
    assert "ORG" == spans[0].tag
    assert "Berlin" == spans[1].text
    assert "LOC" == spans[1].tag

    # all tags
    sentence[0].add_tag("ner", "I-ORG")
    sentence[1].add_tag("ner", "E-ORG")
    sentence[2].add_tag("ner", "aux")
    sentence[3].add_tag("ner", "verb")
    sentence[4].add_tag("ner", "preposition")
    sentence[5].add_tag("ner", "I-LOC")

    spans: List[Span] = sentence.get_spans("ner")
    assert 5 == len(spans)
    assert "Zalando Research" == spans[0].text
    assert "ORG" == spans[0].tag
    assert "Berlin" == spans[4].text
    assert "LOC" == spans[4].tag

    # all weird tags
    sentence[0].add_tag("ner", "I-ORG")
    sentence[1].add_tag("ner", "S-LOC")
    sentence[2].add_tag("ner", "aux")
    sentence[3].add_tag("ner", "B-relation")
    sentence[4].add_tag("ner", "E-preposition")
    sentence[5].add_tag("ner", "S-LOC")

    spans: List[Span] = sentence.get_spans("ner")
    assert 5 == len(spans)
    assert "Zalando" == spans[0].text
    assert "ORG" == spans[0].tag
    assert "Research" == spans[1].text
    assert "LOC" == spans[1].tag
    assert "located in" == spans[3].text
    assert "relation" == spans[3].tag

    sentence = Sentence(
        "A woman was charged on Friday with terrorist offences after three Irish Republican Army mortar bombs were found in a Belfast house , police said . "
    )
    sentence[11].add_tag("ner", "S-MISC")
    sentence[12].add_tag("ner", "B-MISC")
    sentence[13].add_tag("ner", "E-MISC")
    spans: List[Span] = sentence.get_spans("ner")
    assert 2 == len(spans)
    assert "Irish" == spans[0].text
    assert "Republican Army" == spans[1].text

    sentence = Sentence("Zalando Research is located in Berlin .")

    # tags with confidence
    sentence[0].add_tag("ner", "B-ORG", 1.0)
    sentence[1].add_tag("ner", "E-ORG", 0.9)
    sentence[5].add_tag("ner", "S-LOC", 0.5)

    spans: List[Span] = sentence.get_spans("ner", min_score=0.0)

    assert 2 == len(spans)
    assert "Zalando Research" == spans[0].text
    assert "ORG" == spans[0].tag
    assert 0.95 == spans[0].score

    assert "Berlin" == spans[1].text
    assert "LOC" == spans[1].tag
    assert 0.5 == spans[1].score

    spans: List[Span] = sentence.get_spans("ner", min_score=0.6)
    assert 1 == len(spans)

    spans: List[Span] = sentence.get_spans("ner", min_score=0.99)
    assert 0 == len(spans)


def test_token_position_in_sentence():
    sentence = Sentence("I love Berlin .")

    assert 0 == sentence.tokens[0].start_position
    assert 1 == sentence.tokens[0].end_position
    assert 2 == sentence.tokens[1].start_position
    assert 6 == sentence.tokens[1].end_position
    assert 7 == sentence.tokens[2].start_position
    assert 13 == sentence.tokens[2].end_position

    sentence = Sentence(" I love  Berlin.", use_tokenizer=segtok_tokenizer)

    assert 1 == sentence.tokens[0].start_position
    assert 2 == sentence.tokens[0].end_position
    assert 3 == sentence.tokens[1].start_position
    assert 7 == sentence.tokens[1].end_position
    assert 9 == sentence.tokens[2].start_position
    assert 15 == sentence.tokens[2].end_position


def test_sentence_to_dict():
    sentence = Sentence(
        "Zalando Research is   located in Berlin, the capital of Germany.",
        use_tokenizer=True,
    ).add_label('class', 'business')

    # bioes tags
    sentence[0].add_tag("ner", "B-ORG")
    sentence[1].add_tag("ner", "E-ORG")
    sentence[5].add_tag("ner", "S-LOC")
    sentence[10].add_tag("ner", "S-LOC")

    dict = sentence.to_dict("ner")

    assert (
        "Zalando Research is   located in Berlin, the capital of Germany."
        == dict["text"]
    )
    assert "Zalando Research" == dict["entities"][0]["text"]
    assert "Berlin" == dict["entities"][1]["text"]
    assert "Germany" == dict["entities"][2]["text"]
    assert 1 == len(dict["labels"])

    sentence = Sentence(
        "Facebook, Inc. is a company, and Google is one as well.",
        use_tokenizer=True,
    )

    # bioes tags
    sentence[0].add_tag("ner", "B-ORG")
    sentence[1].add_tag("ner", "I-ORG")
    sentence[2].add_tag("ner", "E-ORG")
    sentence[8].add_tag("ner", "S-ORG")

    dict = sentence.to_dict("ner")

    assert "Facebook, Inc. is a company, and Google is one as well." == dict["text"]
    assert "Facebook, Inc." == dict["entities"][0]["text"]
    assert "Google" == dict["entities"][1]["text"]
    assert 0 == len(dict["labels"])
