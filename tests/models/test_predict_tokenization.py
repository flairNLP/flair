from flair.data import Dictionary, Sentence
from flair.embeddings import WordEmbeddings
from flair.models import TokenClassifier
from flair.tokenization import SpaceTokenizer, StaccatoTokenizer


def test_prediction_changes_tokenization():

    # simple dummy model
    dictionary = Dictionary()
    dictionary.add_item("test")
    classifier = TokenClassifier(embeddings=WordEmbeddings("turian"),
                                 label_dictionary=dictionary,
                                 label_type='test')

    print(classifier)

    # before and after tokenizer
    tokenizer_A = SpaceTokenizer()
    tokenizer_B = StaccatoTokenizer()

    # init two sentences, trigger tokenization in sentence_1 but not in sentence_2
    sentence_1 = Sentence("I love Berlin.", use_tokenizer=tokenizer_A)
    sentence_1[0] # trigger tokenization
    sentence_2 = Sentence("I love New York.", use_tokenizer=tokenizer_A)

    # classifier gets tokenizer B
    classifier._tokenizer = tokenizer_B

    # predict for the two sentences
    classifier.predict([sentence_1, sentence_2])

    # assert that the length of the two sentences reflects the new tokenizer
    assert len(sentence_1) == 4
    assert len(sentence_2) == 5

    # # assert that entities are correctly found
    # assert
