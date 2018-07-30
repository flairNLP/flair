import shutil

import os

from flair.data import Sentence, Token, Dictionary, TaggedCorpus


def test_get_head():
    token1 = Token('I', 0)
    token2 = Token('love', 1, 0)
    token3 = Token('Berlin', 2, 1)

    sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)

    assert(token2 == token3.get_head())
    assert(token1 == token2.get_head())
    assert(None == token1.get_head())


def test_create_sentence_without_tokenizer():
    sentence = Sentence('I love Berlin.')

    assert(3 == len(sentence.tokens))
    assert('I' == sentence.tokens[0].text)
    assert('love' == sentence.tokens[1].text)
    assert('Berlin.' == sentence.tokens[2].text)


def test_create_sentence_with_tokenizer():
    sentence = Sentence('I love Berlin.', use_tokenizer=True)

    assert (4 == len(sentence.tokens))
    assert ('I' == sentence.tokens[0].text)
    assert ('love' == sentence.tokens[1].text)
    assert ('Berlin' == sentence.tokens[2].text)
    assert ('.' == sentence.tokens[3].text)


def test_sentence_to_plain_string():
    sentence = Sentence('I love Berlin.', use_tokenizer=True)

    assert ('I love Berlin .' == sentence.to_plain_string())


def test_sentence_to_tagged_string():
    token1 = Token('I', 0)
    token2 = Token('love', 1, 0)
    token3 = Token('Berlin', 2, 1)
    token3.add_tag('ner', 'LOC')

    sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)

    assert ('I love Berlin <LOC>' == sentence.to_tagged_string())


def test_dictionary_get_items_with_unk():
    dictionary = Dictionary()

    dictionary.add_item('class_1')
    dictionary.add_item('class_2')
    dictionary.add_item('class_3')

    items = dictionary.get_items()

    assert(4 == len(items))
    assert('<unk>' == items[0])
    assert('class_1' == items[1])
    assert('class_2' == items[2])
    assert('class_3' == items[3])


def test_dictionary_get_items_without_unk():
    dictionary = Dictionary(add_unk=False)

    dictionary.add_item('class_1')
    dictionary.add_item('class_2')
    dictionary.add_item('class_3')

    items = dictionary.get_items()

    assert(3 == len(items))
    assert('class_1' == items[0])
    assert('class_2' == items[1])
    assert('class_3' == items[2])


def test_dictionary_get_idx_for_item():
    dictionary = Dictionary(add_unk=False)

    dictionary.add_item('class_1')
    dictionary.add_item('class_2')
    dictionary.add_item('class_3')

    idx = dictionary.get_idx_for_item('class_2')

    assert(1 == idx)


def test_dictionary_get_item_for_index():
    dictionary = Dictionary(add_unk=False)

    dictionary.add_item('class_1')
    dictionary.add_item('class_2')
    dictionary.add_item('class_3')

    item = dictionary.get_item_for_index(0)

    assert('class_1' == item)


def test_dictionary_save_and_load():
    dictionary = Dictionary(add_unk=False)

    dictionary.add_item('class_1')
    dictionary.add_item('class_2')
    dictionary.add_item('class_3')

    file_path = 'dictionary.txt'

    dictionary.save(file_path)
    loaded_dictionary = dictionary.load_from_file(file_path)

    assert(len(dictionary) == len(loaded_dictionary))
    assert(len(dictionary.get_items()) == len(loaded_dictionary.get_items()))

    # clean up file
    os.remove(file_path)


def test_tagged_corpus_get_all_sentences():
    train_sentence = Sentence("I'm used in training.", use_tokenizer=True)
    dev_sentence = Sentence("I'm a dev sentence.", use_tokenizer=True)
    test_sentence = Sentence("I will be only used for testing.", use_tokenizer=True)

    corpus = TaggedCorpus([train_sentence], [dev_sentence], [test_sentence])

    all_sentences = corpus.get_all_sentences()

    assert(3 == len(all_sentences))


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence('used in training. training is cool.', use_tokenizer=True)

    corpus = TaggedCorpus([train_sentence], [], [])

    vocab = corpus.make_vocab_dictionary(max_tokens=2, min_freq=-1)

    assert(3 == len(vocab))
    assert('<unk>' in vocab.get_items())
    assert('training' in vocab.get_items())
    assert('.' in vocab.get_items())

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=-1)

    assert(7 == len(vocab))

    vocab = corpus.make_vocab_dictionary(max_tokens=-1, min_freq=2)

    assert(3 == len(vocab))
    assert('<unk>' in vocab.get_items())
    assert('training' in vocab.get_items())
    assert('.' in vocab.get_items())


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence('sentence 1', labels=['class_1'])
    sentence_2 = Sentence('sentence 2', labels=['class_2'])
    sentence_3 = Sentence('sentence 3', labels=['class_1'])

    corpus = TaggedCorpus([sentence_1, sentence_2, sentence_3], [], [])

    label_dict = corpus.make_label_dictionary()

    assert(2 == len(label_dict))
    assert('<unk>' not in label_dict.get_items())
    assert('class_1' in label_dict.get_items())
    assert('class_2' in label_dict.get_items())


def test_tagged_corpus_statistics():
    train_sentence = Sentence('I love Berlin.', labels=['class_1'], use_tokenizer=True)
    dev_sentence = Sentence('The sun is shining.', labels=['class_2'], use_tokenizer=True)
    test_sentence = Sentence('Berlin is sunny.', labels=['class_1'], use_tokenizer=True)

    class_to_count_dict = TaggedCorpus._get_classes_to_count([train_sentence, dev_sentence, test_sentence])

    assert('class_1' in class_to_count_dict)
    assert('class_2' in class_to_count_dict)
    assert(2 == class_to_count_dict['class_1'])
    assert(1 == class_to_count_dict['class_2'])

    tokens_in_sentences = TaggedCorpus._get_tokens_per_sentence([train_sentence, dev_sentence, test_sentence])

    assert(3 == len(tokens_in_sentences))
    assert(4 == tokens_in_sentences[0])
    assert(5 == tokens_in_sentences[1])
    assert(4 == tokens_in_sentences[2])


def test_tagged_corpus_downsample():
    sentence = Sentence('I love Berlin.', labels=['class_1'], use_tokenizer=True)

    corpus = TaggedCorpus(
        [sentence, sentence, sentence, sentence, sentence, sentence, sentence, sentence, sentence, sentence], [], [])

    assert(10 == len(corpus.train))

    corpus.downsample(percentage=0.3, only_downsample_train=True)

    assert(3 == len(corpus.train))