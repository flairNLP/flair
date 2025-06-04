from flair.data import Dictionary, Sentence
from flair.embeddings import WordEmbeddings
from flair.models import TokenClassifier, SequenceTagger
from flair.tokenization import SpaceTokenizer, StaccatoTokenizer


def test_prediction_changes_tokenization():

    # simple dummy model
    dictionary = Dictionary()
    dictionary.add_item("test")
    classifier = TokenClassifier(embeddings=WordEmbeddings("turian"), label_dictionary=dictionary, label_type="test")

    # before and after tokenizer
    tokenizer_A = SpaceTokenizer()
    tokenizer_B = StaccatoTokenizer()

    # init two sentences, trigger tokenization in sentence_1 but not in sentence_2 and sentence_3
    sentence_1 = Sentence("I love Berlin.", use_tokenizer=tokenizer_A)
    assert sentence_1[0].text == "I"  # trigger tokenization
    assert len(sentence_1) == 3

    sentence_2 = Sentence("I love New York.", use_tokenizer=tokenizer_A)
    sentence_3 = Sentence("I love Den Haag.", use_tokenizer=tokenizer_A)

    # classifier predicts without any tokenizer for the first two sentences
    classifier.predict([sentence_1, sentence_2])
    assert len(sentence_1) == 3
    assert len(sentence_2) == 4

    # now classifier gets tokenizer B
    classifier._tokenizer = tokenizer_B

    # predict for all three sentences
    classifier.predict([sentence_1, sentence_2, sentence_3])

    # assert that the length of the two sentences reflects the new tokenizer
    assert len(sentence_1) == 4
    assert len(sentence_2) == 5
    assert len(sentence_3) == 5


def test_prediction_changes_tokenization_sequence_tagger():

    # NER model
    classifier = SequenceTagger.load("ner-fast")

    # before and after tokenizer
    tokenizer_A = SpaceTokenizer()
    tokenizer_B = StaccatoTokenizer()

    # init two sentences, trigger tokenization in sentence_1 but not in sentence_2 and sentence_3
    sentence_1 = Sentence("I love Berlin.", use_tokenizer=tokenizer_A)
    assert sentence_1[0].text == "I"  # trigger tokenization
    assert len(sentence_1) == 3

    sentence_2 = Sentence("I love New York.", use_tokenizer=tokenizer_A)
    sentence_3 = Sentence("I love Den Haag.", use_tokenizer=tokenizer_A)

    # classifier predicts without any tokenizer for the first two sentences
    classifier.predict([sentence_1, sentence_2])
    assert len(sentence_1) == 3
    assert len(sentence_1.get_spans()) == 0
    assert len(sentence_2) == 4

    # now classifier gets tokenizer B
    classifier._tokenizer = tokenizer_B

    # predict for all three sentences
    classifier.predict([sentence_1, sentence_2, sentence_3])

    # assert that the length of the two sentences reflects the new tokenizer
    assert len(sentence_1) == 4
    assert len(sentence_1.get_spans()) == 1
    assert len(sentence_2) == 5
    assert len(sentence_3) == 5

    # one more sentence without tokenizer
    sentence_4 = Sentence(["I", "love", "Berlin."])
    classifier._tokenizer = None
    classifier.predict([sentence_4])
    assert len(sentence_4) == 3
    classifier._tokenizer = StaccatoTokenizer()
    classifier.predict([sentence_4])
    assert len(sentence_4) == 4
