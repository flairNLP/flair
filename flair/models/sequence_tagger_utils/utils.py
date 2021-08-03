import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import flair
from flair.data import Label

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


def get_tags_tensor(sentences, tag_dictionary, tag_type, using_crf):
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
    # Original tags can be recovered by tag_list % len(tag_dictionary) - see if condition below
    tag_list = list(map(lambda sentence:
                        [tag_dictionary.get_idx_for_item(START_TAG) * len(tag_dictionary) + sentence[0]]
                        + [sentence[index - 1] * len(tag_dictionary) + sentence[index] for index in range(1, len(sentence))]
                        , tag_list))

    if not using_crf:
        tag_list = [list(map(lambda token: token % len(tag_dictionary), current_tag_list)) for current_tag_list in tag_list]

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


def obtain_labels(features: torch.Tensor, lengths: torch.Tensor, tag_dictionary):
    """
    :param features: torch.Tensor
    :param lengths: torch Length object
    :param tag_dictionary: Dictionary containing mapping between IDs and Labels
    :return tags: List containing all decoded labels
    Obtain labels by applying softmax function. Alternative to CRF viterbi decoding.
    """
    softmax_batch = torch.nn.functional.softmax(features, dim=2).cpu()
    scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
    feature = zip(softmax_batch, scores_batch, prediction_batch)

    tags = []
    for feats, length in zip(feature, lengths.values.tolist()):
        softmax, score, prediction = feats
        confidences = score[:length].tolist()
        tag_seq = prediction[:length].tolist()

        tags.append(
            [
                Label(tag_dictionary.get_item_for_index(tag), conf)
                for conf, tag in zip(confidences, tag_seq)
            ]
        )

    return tags