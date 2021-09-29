import flair
import os
import pytest

from typing import List

from flair.data import (
    Sentence,
    Label,
    Token,
    Dictionary,
    Corpus,
    Span
)
from flair.tokenization import (
    SpacyTokenizer,
    SegtokTokenizer,
    JapaneseTokenizer,
    TokenizerWrapper,
    SciSpacyTokenizer,
    SegtokSentenceSplitter,
    NoSentenceSplitter,
    TagSentenceSplitter,
    NewlineSentenceSplitter,
    SpacySentenceSplitter,
    SciSpacySentenceSplitter
)


def no_op_tokenizer(text: str) -> List[Token]:
    return [Token(text, idx=0, start_position=0)]


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
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=False)

    assert 3 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I" == sentence.tokens[0].text
    assert 2 == sentence.tokens[1].start_pos
    assert "love" == sentence.tokens[1].text
    assert 7 == sentence.tokens[2].start_pos
    assert "Berlin." == sentence.tokens[2].text


def test_create_sentence_with_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=True)

    assert 4 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I" == sentence.tokens[0].text
    assert 2 == sentence.tokens[1].start_pos
    assert "love" == sentence.tokens[1].text
    assert 7 == sentence.tokens[2].start_pos
    assert "Berlin" == sentence.tokens[2].text
    assert 13 == sentence.tokens[3].start_pos
    assert "." == sentence.tokens[3].text


def test_create_sentence_with_newline():
    sentence: Sentence = Sentence(["I", "\t", "ich", "\n", "you", "\t", "du", "\n"])
    assert 8 == len(sentence.tokens)
    assert "\n" == sentence.tokens[3].text

    sentence: Sentence = Sentence("I \t ich \n you \t du \n", use_tokenizer=False)
    assert 8 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "\n" == sentence.tokens[3].text


def test_create_sentence_with_custom_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=TokenizerWrapper(no_op_tokenizer))
    assert 1 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I love Berlin." == sentence.tokens[0].text


def test_create_sentence_with_callable():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=no_op_tokenizer)
    assert 1 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I love Berlin." == sentence.tokens[0].text


@pytest.mark.skip(reason="SpacyTokenizer needs optional requirements, so we skip the test by default")
def test_create_sentence_with_spacy_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SpacyTokenizer("en_core_sci_sm"))

    assert 4 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I" == sentence.tokens[0].text
    assert 2 == sentence.tokens[1].start_pos
    assert "love" == sentence.tokens[1].text
    assert 7 == sentence.tokens[2].start_pos
    assert "Berlin" == sentence.tokens[2].text
    assert 13 == sentence.tokens[3].start_pos
    assert "." == sentence.tokens[3].text


def test_create_sentence_using_japanese_tokenizer():
    sentence: Sentence = Sentence("私はベルリンが好き", use_tokenizer=JapaneseTokenizer("janome"))

    assert 5 == len(sentence.tokens)
    assert "私" == sentence.tokens[0].text
    assert "は" == sentence.tokens[1].text
    assert "ベルリン" == sentence.tokens[2].text
    assert "が" == sentence.tokens[3].text
    assert "好き" == sentence.tokens[4].text


@pytest.mark.skip(reason="SciSpacyTokenizer need optional requirements, so we skip the test by default")
def test_create_sentence_using_scispacy_tokenizer():
    sentence: Sentence = Sentence(
        "Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron",
        use_tokenizer=SciSpacyTokenizer()
    )

    assert 13 == len(sentence.tokens)
    assert "Spinal" == sentence.tokens[0].text
    assert "and" == sentence.tokens[1].text
    assert "bulbar" == sentence.tokens[2].text
    assert "muscular" == sentence.tokens[3].text
    assert "atrophy" == sentence.tokens[4].text
    assert "(" == sentence.tokens[5].text
    assert "SBMA" == sentence.tokens[6].text
    assert ")" == sentence.tokens[7].text
    assert "is" == sentence.tokens[8].text
    assert "an" == sentence.tokens[9].text
    assert "inherited" == sentence.tokens[10].text
    assert "motor" == sentence.tokens[11].text
    assert "neuron" == sentence.tokens[12].text

    assert 0 == sentence.tokens[0].start_pos
    assert 7 == sentence.tokens[1].start_pos
    assert 11 == sentence.tokens[2].start_pos
    assert 18 == sentence.tokens[3].start_pos
    assert 27 == sentence.tokens[4].start_pos
    assert 35 == sentence.tokens[5].start_pos
    assert 36 == sentence.tokens[6].start_pos
    assert 40 == sentence.tokens[7].start_pos
    assert 42 == sentence.tokens[8].start_pos
    assert 45 == sentence.tokens[9].start_pos
    assert 48 == sentence.tokens[10].start_pos
    assert 58 == sentence.tokens[11].start_pos
    assert 64 == sentence.tokens[12].start_pos

    assert True == sentence.tokens[4].whitespace_after
    assert False == sentence.tokens[5].whitespace_after
    assert False == sentence.tokens[6].whitespace_after
    assert True == sentence.tokens[7].whitespace_after


def test_segtok_sentence_splitter():
    segtok_splitter = SegtokSentenceSplitter()
    sentences = segtok_splitter.split("I love Berlin. Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 6

    segtok_splitter = SegtokSentenceSplitter(tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = segtok_splitter.split("I love Berlin. Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 1


def test_no_sentence_splitter():
    no_splitter = NoSentenceSplitter()
    sentences = no_splitter.split("I love Berlin")
    assert len(sentences) == 1
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 3

    no_splitter = NoSentenceSplitter(TokenizerWrapper(no_op_tokenizer))
    sentences = no_splitter.split("I love Berlin")
    assert len(sentences) == 1
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1


def test_tag_sentence_splitter():
    tag_splitter = TagSentenceSplitter(tag="#!")

    sentences = tag_splitter.split("I love Berlin#!Me too")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 3
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 2

    tag_splitter = TagSentenceSplitter(tag="#!", tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = tag_splitter.split("I love Berlin#!Me too")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 1

    sentences = tag_splitter.split("I love Berlin Me too")
    assert len(sentences) == 1

    sentences = tag_splitter.split("I love Berlin#!#!Me too")
    assert len(sentences) == 2

    sentences = tag_splitter.split("I love Berl#! #!inMe too")
    assert len(sentences) == 2


def test_newline_sentence_splitter():
    newline_splitter = NewlineSentenceSplitter()

    sentences = newline_splitter.split("I love Berlin\nMe too")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 3
    assert sentences[0].start_pos == 0
    assert len(sentences[1].tokens) == 2

    newline_splitter = NewlineSentenceSplitter(tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = newline_splitter.split("I love Berlin\nMe too")
    assert len(sentences) == 2
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 14
    assert len(sentences[1].tokens) == 1

    sentences = newline_splitter.split("I love Berlin Me too")
    assert len(sentences) == 1

    sentences = newline_splitter.split("I love Berlin\n\nMe too")
    assert len(sentences) == 2

    sentences = newline_splitter.split("I love Berlin\n \nMe too")
    assert len(sentences) == 2


@pytest.mark.skip(reason="SpacySentenceSplitter need optional requirements, so we skip the test by default")
def test_spacy_sentence_splitter():
    spacy_splitter = SpacySentenceSplitter("en_core_sci_sm")

    sentences = spacy_splitter.split("This a sentence. And here is another one.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 17
    assert len(sentences[1].tokens) == 6

    sentences = spacy_splitter.split("VF inhibits something. ACE-dependent (GH+) issuses too.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 23
    assert len(sentences[1].tokens) == 7

    spacy_splitter = SpacySentenceSplitter("en_core_sci_sm", tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = spacy_splitter.split("This a sentence. And here is another one.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 17
    assert len(sentences[1].tokens) == 1


@pytest.mark.skip(reason="SciSpacySentenceSplitter need optional requirements, so we skip the test by default")
def test_scispacy_sentence_splitter():
    scispacy_splitter = SciSpacySentenceSplitter()
    sentences = scispacy_splitter.split("VF inhibits something. ACE-dependent (GH+) issuses too.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 23
    assert len(sentences[1].tokens) == 9


def test_problem_sentences():
    text = "so out of the norm ❤ ️ enjoyed every moment️"
    sentence = Sentence(text)
    assert len(sentence) == 9

    text = "equivalently , accumulating the logs as :( 6 ) sl = 1N ∑ t = 1Nlogp ( Ll | xt ​ , θ ) where " \
           "p ( Ll | xt ​ , θ ) represents the class probability output"
    sentence = Sentence(text)
    assert len(sentence) == 37

    text = "This guy needs his own show on Discivery Channel ! ﻿"
    sentence = Sentence(text)
    assert len(sentence) == 10

    text = "n't have new vintages."
    sentence = Sentence(text, use_tokenizer=True)
    assert len(sentence) == 5


def test_token_indices():
    text = ":    nation on"
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = ":    nation on"
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()

    text = "I love Berlin."
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = 'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = 'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()


def test_create_sentence_with_segtoktokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert 4 == len(sentence.tokens)
    assert "I" == sentence.tokens[0].text
    assert "love" == sentence.tokens[1].text
    assert "Berlin" == sentence.tokens[2].text
    assert "." == sentence.tokens[3].text


def test_sentence_to_plain_string():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert "I love Berlin ." == sentence.to_tokenized_string()


def test_sentence_to_real_string(tasks_base_path):
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())
    assert "I love Berlin." == sentence.to_plain_string()

    corpus = flair.datasets.NER_GERMAN_GERMEVAL(base_path=tasks_base_path)

    sentence = corpus.train[0]
    sentence.infer_space_after()
    assert (
            'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
            == sentence.to_tokenized_string()
    )
    assert (
            'Schartau sagte dem "Tagesspiegel" vom Freitag, Fischer sei "in einer Weise aufgetreten, die alles andere als überzeugend war".'
            == sentence.to_plain_string()
    )

    sentence = corpus.train[1]
    sentence.infer_space_after()
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
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

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
    train_sentence = Sentence("I'm used in training.", use_tokenizer=SegtokTokenizer())
    dev_sentence = Sentence("I'm a dev sentence.", use_tokenizer=SegtokTokenizer())
    test_sentence = Sentence(
        "I will be only used for testing.", use_tokenizer=SegtokTokenizer()
    )

    corpus: Corpus = Corpus([train_sentence], [dev_sentence], [test_sentence])

    all_sentences = corpus.get_all_sentences()

    assert 3 == len(all_sentences)


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence(
        "used in training. training is cool.", use_tokenizer=SegtokTokenizer()
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

    assert 3.2 == label.score
    assert "class_1" == label.value

    label.score = 0.2

    assert 0.2 == label.score


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence("sentence 1").add_label('label', 'class_1')

    sentence_2 = Sentence("sentence 2").add_label('label', 'class_2')

    sentence_3 = Sentence("sentence 3").add_label('label', 'class_1')

    corpus: Corpus = Corpus([sentence_1, sentence_2, sentence_3], [], [])

    label_dict = corpus.make_label_dictionary('label')

    assert 3 == len(label_dict)
    assert "<unk>" in label_dict.get_items()
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
        use_tokenizer=SegtokTokenizer(),
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


def test_classification_corpus_multi_labels_without_negative_examples(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "multi_class_negative_examples",
                                                 allow_examples_without_labels=False)
    assert len(corpus.train) == 7
    assert len(corpus.dev) == 4
    assert len(corpus.test) == 5


def test_classification_corpus_multi_labels_with_negative_examples(tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "multi_class_negative_examples",
                                                 allow_examples_without_labels=True)
    assert len(corpus.train) == 8
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 6


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

    sentence = Sentence(" I love  Berlin.", use_tokenizer=SegtokTokenizer())

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


def test_pretokenized():
    pretoks = ['The', 'grass', 'is', 'green', '.']
    sent = Sentence(pretoks)
    for i, token in enumerate(sent):
        assert token.text == pretoks[i]


@pytest.fixture
def sentence_with_relations():
    # city single-token, person and company multi-token
    sentence = Sentence("Person A , born in city , works for company B .")

    sentence[0].add_tag("ner", "B-Peop")
    sentence[1].add_tag("ner", "I-Peop")
    sentence[1].add_tag("relation", "['Born_In', 'Works_For']")
    sentence[1].add_tag("relation_dep", "[5, 10]")
    sentence[5].add_tag("ner", "B-Loc")
    sentence[9].add_tag("ner", "B-Org")
    sentence[10].add_tag("ner", "I-Org")
    for i in range(len(sentence)):
        if i != 1:
            sentence[i].add_tag("relation", "['N']")
            sentence[i].add_tag("relation_dep", f"[{i}]")

    return sentence