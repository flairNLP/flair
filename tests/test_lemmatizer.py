import pytest

from flair.models import Lemmatizer
from flair.data import Sentence, Dictionary


@pytest.mark.integration
def test_words_to_char_indices(sentence: Sentence, dic: Dictionary = None):

    lemmatizer = Lemmatizer(char_dict=dic)

    string_list = sentence.to_tokenized_string().split()

    print('With end symbol, without start symbol, padding in front')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=True, start_symbol= False, padding_in_front = True).size())
    print('-------------------------------------------------------')
    print('With end symbol, without start symbol, padding in back')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=True, start_symbol= False, padding_in_front = False))
    print('-------------------------------------------------------')
    print('With end symbol, with start symbol, padding in front')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=True, start_symbol= True, padding_in_front = True))
    print('-------------------------------------------------------')
    print('With end symbol, with start symbol, padding in back')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=True, start_symbol= True, padding_in_front = False))
    print('-------------------------------------------------------')
    print('Without end symbol, without start symbol, padding in front')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=False, start_symbol= False, padding_in_front = True))
    print('-------------------------------------------------------')
    print('Without end symbol, without start symbol, padding in back')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=False, start_symbol= False, padding_in_front = False))
    print('-------------------------------------------------------')
    print('Without end symbol, with start symbol, padding in front')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=False, start_symbol= True, padding_in_front = True))
    print('-------------------------------------------------------')
    print('Without end symbol, with start symbol, padding in back')
    print(lemmatizer.words_to_char_indices(string_list, end_symbol=False, start_symbol= True, padding_in_front = False))




