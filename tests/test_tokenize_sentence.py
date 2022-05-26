from typing import List

import pytest

import flair
from flair.data import Sentence, Token
from flair.tokenization import (
    JapaneseTokenizer,
    NewlineSentenceSplitter,
    NoSentenceSplitter,
    SciSpacySentenceSplitter,
    SciSpacyTokenizer,
    SegtokSentenceSplitter,
    SegtokTokenizer,
    SpaceTokenizer,
    SpacySentenceSplitter,
    SpacyTokenizer,
    TagSentenceSplitter,
    TokenizerWrapper,
)


def test_create_sentence_on_empty_string():
    sentence: Sentence = Sentence("")
    assert 0 == len(sentence.tokens)


def test_create_sentence_with_newline():
    sentence: Sentence = Sentence(["I", "\t", "ich", "\n", "you", "\t", "du", "\n"])
    assert 8 == len(sentence.tokens)
    assert "\n" == sentence.tokens[3].text

    sentence: Sentence = Sentence("I \t ich \n you \t du \n", use_tokenizer=False)
    assert 8 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "\n" == sentence.tokens[3].text


def test_create_sentence_with_extra_whitespace():
    sentence: Sentence = Sentence("I  love Berlin .")

    assert 4 == len(sentence.tokens)
    assert "I" == sentence.get_token(1).text
    assert "love" == sentence.get_token(2).text
    assert "Berlin" == sentence.get_token(3).text
    assert "." == sentence.get_token(4).text


def test_create_sentence_difficult_encoding():
    text = "so out of the norm ❤ ️ enjoyed every moment️"
    sentence = Sentence(text)
    assert len(sentence) == 9

    text = (
        "equivalently , accumulating the logs as :( 6 ) sl = 1N ∑ t = 1Nlogp "
        "( Ll | xt ​ , θ ) where "
        "p ( Ll | xt ​ , θ ) represents the class probability output"
    )
    sentence = Sentence(text)
    assert len(sentence) == 37

    text = "This guy needs his own show on Discivery Channel ! ﻿"
    sentence = Sentence(text)
    assert len(sentence) == 10

    text = "n't have new vintages."
    sentence = Sentence(text, use_tokenizer=True)
    assert len(sentence) == 5


def test_create_sentence_word_by_word():
    token1: Token = Token("Munich")
    token2: Token = Token("and")
    token3: Token = Token("Berlin")
    token4: Token = Token("are")
    token5: Token = Token("nice")

    sentence: Sentence = Sentence([])

    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)
    sentence.add_token(token4)
    sentence.add_token(token5)

    sentence.add_token("cities")
    sentence.add_token(Token("."))

    assert "Munich and Berlin are nice cities ." == sentence.to_tokenized_string()


def test_create_sentence_pretokenized():
    pretoks = ["The", "grass", "is", "green", "."]
    sent = Sentence(pretoks)
    for i, token in enumerate(sent):
        assert token.text == pretoks[i]


def test_create_sentence_without_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=False)

    assert 3 == len(sentence.tokens)
    assert 0 == sentence.tokens[0].start_pos
    assert "I" == sentence.tokens[0].text
    assert 2 == sentence.tokens[1].start_pos
    assert "love" == sentence.tokens[1].text
    assert 7 == sentence.tokens[2].start_pos
    assert "Berlin." == sentence.tokens[2].text


def test_create_sentence_with_default_tokenizer():
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


def test_create_sentence_with_segtok():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert 4 == len(sentence.tokens)
    assert "I" == sentence.tokens[0].text
    assert "love" == sentence.tokens[1].text
    assert "Berlin" == sentence.tokens[2].text
    assert "." == sentence.tokens[3].text


def test_create_sentence_with_custom_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=TokenizerWrapper(no_op_tokenizer))
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


@pytest.mark.skip(reason="SciSpacyTokenizer need optional requirements, " "so we skip the test by default")
def test_create_sentence_using_scispacy_tokenizer():
    sentence: Sentence = Sentence(
        "Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron",
        use_tokenizer=SciSpacyTokenizer(),
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

    assert sentence.tokens[4].whitespace_after == 1
    assert sentence.tokens[5].whitespace_after != 1
    assert sentence.tokens[6].whitespace_after != 1
    assert sentence.tokens[7].whitespace_after == 1


def test_split_text_segtok():
    segtok_splitter = SegtokSentenceSplitter()
    sentences = segtok_splitter.split("I love Berlin. " "Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 6

    segtok_splitter = SegtokSentenceSplitter(tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = segtok_splitter.split("I love Berlin. " "Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 15
    assert len(sentences[1].tokens) == 1


def test_split_text_nosplit():
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


def test_split_text_on_tag():
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


def test_split_text_on_newline():
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


@pytest.mark.skip(reason="SpacySentenceSplitter need optional requirements, " "so we skip the test by default")
def test_split_text_spacy():
    spacy_splitter = SpacySentenceSplitter("en_core_sci_sm")

    sentences = spacy_splitter.split("This a sentence. " "And here is another one.")
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
    sentences = spacy_splitter.split("This a sentence. " "And here is another one.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_pos == 17
    assert len(sentences[1].tokens) == 1


@pytest.mark.skip(reason="SciSpacySentenceSplitter need optional requirements, " "so we skip the test by default")
def test_split_text_scispacy():
    scispacy_splitter = SciSpacySentenceSplitter()
    sentences = scispacy_splitter.split("VF inhibits something. ACE-dependent (GH+) issuses too.")
    assert len(sentences) == 2
    assert sentences[0].start_pos == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_pos == 23
    assert len(sentences[1].tokens) == 9


def test_print_sentence_tokenized():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert "I love Berlin ." == sentence.to_tokenized_string()


def test_print_original_text():
    text = ":    nation on"
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = ":    nation on"
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()

    text = "I love Berlin."
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " '
        "in einer Weise aufgetreten , die alles andere als überzeugend "
        'war " .'
    )
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " '
        "in einer Weise aufgetreten , die alles andere als überzeugend "
        'war " .'
    )
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()


def test_print_sentence_plain(tasks_base_path):
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())
    assert "I love Berlin." == sentence.to_plain_string()

    corpus = flair.datasets.NER_GERMAN_GERMEVAL(base_path=tasks_base_path)

    sentence = corpus.train[0]
    sentence.infer_space_after()
    assert (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in '
        "einer Weise aufgetreten , "
        'die alles andere als überzeugend war " .' == sentence.to_tokenized_string()
    )
    assert (
        'Schartau sagte dem "Tagesspiegel" vom Freitag, Fischer sei "in einer '
        "Weise aufgetreten, die "
        'alles andere als überzeugend war".' == sentence.to_plain_string()
    )

    sentence = corpus.train[1]
    sentence.infer_space_after()
    assert (
        "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als "
        "Möbelvertreter , als er einen fliegenden Händler aus dem Libanon traf ." == sentence.to_tokenized_string()
    )
    assert (
        "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als "
        "Möbelvertreter, als er einen fliegenden Händler aus dem Libanon traf." == sentence.to_plain_string()
    )


def test_infer_space_after():
    sentence: Sentence = Sentence([])
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
        _ = sentence[4]


def test_token_positions_when_creating_with_tokenizer():
    sentence = Sentence("I love Berlin .", use_tokenizer=SpaceTokenizer())

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


def test_token_positions_when_creating_word_by_word():
    sentence: Sentence = Sentence([])
    sentence.add_token(Token("I"))
    sentence.add_token("love")
    sentence.add_token("Berlin")
    sentence.add_token(".")

    assert 0 == sentence.tokens[0].start_position
    assert 1 == sentence.tokens[0].end_position
    assert 2 == sentence.tokens[1].start_position
    assert 6 == sentence.tokens[1].end_position
    assert 7 == sentence.tokens[2].start_position
    assert 13 == sentence.tokens[2].end_position


def no_op_tokenizer(text: str) -> List[str]:
    return [text]
