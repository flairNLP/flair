import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import flair

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


def get_tags_tensor(sentences, tag_dictionary, tag_type):
    """
    Transforms a given batch of sentences into tag tensors.
    :return: torch.Tensor
    """
    # Transfrom each sentences into list of tokens
    # i.e. [[token_1, token_2], [token_1, token_2, token_3], ...]
    token_list = list(map(lambda sentence: sentence.tokens, sentences))

    # Transform token_list from each sentences into the respective tag id from tag_dictionary
    # and add the STOP_TAG after it
    # i.e. [[1,2,11], [1,2,3,11], ...] if STOP_TAG has ID = 11
    tag_list = list(map(lambda sentence:
                        list(map(lambda token:
                                 tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value), sentence))
                                 + [tag_dictionary.get_idx_for_item(STOP_TAG)]
                        , token_list))

    # Following transformation consists of two parts:
    # (1) add transition from start tag to each sequence i.e. [1,2,3,11] becomes [10,1,2,3,11] if start tag has ID = 10
    # (2) transform tag_list into matrix indices from CRF scores
    # i.e. consider 12x12 crf score matrix (emission scores + transition matrix) and our sequence is [10,1,11]
    # then, considering our first tag, we're looking for transition from 10 to 1 + emission_score for 1
    # In our unrolled crf score matrix the equals ID = 121 due to
    # row_index (from tag 10) * length of tagset (12) + column index (to tag 1)
    # Original tags can be recovered by tag_list % len(tag_dictionary)
    tag_list = list(map(lambda sentence:
                        [tag_dictionary.get_idx_for_item(START_TAG) * len(tag_dictionary) + sentence[0]]
                        + [sentence[index - 1] * len(tag_dictionary) + sentence[index] for index in range(1, len(sentence))]
                        , tag_list))

    # Transform list to a list of LongTensor
    tag_list_as_tensor = list(map(lambda tags: torch.LongTensor(tags).to(flair.device), tag_list))

    # pad tag_list so that we return a tensor of shape (batch_size, seq len + 1) since we're added transition to end tag
    padded_tag_tensor = pad_sequence(tag_list_as_tensor, batch_first=True)

    return padded_tag_tensor


def init_stop_tag_embedding(embedding_length):
    """
    Initializes a stop embedding.
    """
    bias = np.sqrt(3.0 / embedding_length)
    return nn.init.uniform_(torch.FloatTensor(embedding_length), -bias, bias).to(flair.device)