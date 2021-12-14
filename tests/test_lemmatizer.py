import torch

import flair
from flair.data import Sentence
from flair.models import Lemmatizer


def test_words_to_char_indices():
    sentence = Sentence("Hello look what a beautiful day!")

    lemmatizer = Lemmatizer()  # lemmatizer uses standard char dictionary

    d = lemmatizer.dummy_index
    e = lemmatizer.end_index
    s = lemmatizer.start_index

    string_list = sentence.to_tokenized_string().split()

    # With end symbol, without start symbol, padding in front
    target = torch.tensor(
        [
            [d, d, d, d, 55, 5, 15, 15, 12, e],
            [d, d, d, d, d, 15, 12, 12, 28, e],
            [d, d, d, d, d, 23, 13, 9, 8, e],
            [d, d, d, d, d, d, d, d, 9, e],
            [24, 5, 9, 16, 8, 7, 22, 16, 15, e],
            [d, d, d, d, d, d, 14, 9, 27, e],
            [d, d, d, d, d, d, d, d, 76, e],
        ],
        dtype=torch.long,
    ).to(flair.device)
    out = lemmatizer.words_to_char_indices(string_list, end_symbol=True, start_symbol=False, padding_in_front=True)
    assert torch.equal(target, out)

    # Without end symbol, with start symbol, padding in back
    target = torch.tensor(
        [
            [s, 55, 5, 15, 15, 12, d, d, d, d],
            [s, 15, 12, 12, 28, d, d, d, d, d],
            [s, 23, 13, 9, 8, d, d, d, d, d],
            [s, 9, d, d, d, d, d, d, d, d],
            [s, 24, 5, 9, 16, 8, 7, 22, 16, 15],
            [s, 14, 9, 27, d, d, d, d, d, d],
            [s, 76, d, d, d, d, d, d, d, d],
        ],
        dtype=torch.long,
    ).to(flair.device)
    out = lemmatizer.words_to_char_indices(string_list, end_symbol=False, start_symbol=True, padding_in_front=False)
    assert torch.equal(target, out)

    # Without end symbol, without start symbol, padding in front
    assert lemmatizer.words_to_char_indices(
        string_list, end_symbol=False, start_symbol=False, padding_in_front=True
    ).size() == (7, 9)
