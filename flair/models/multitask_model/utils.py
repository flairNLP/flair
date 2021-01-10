import torch

import flair
from flair.data import List

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

def get_tag_list(sentences, tag_dictionary, tag_type):
    tag_list = list()
    for s_id, sentence in enumerate(sentences):
        # get the tags in this sentence
        tag_idx: List[int] = [
            tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value)
            for token in sentence
        ]
        # add tags as tensor
        tag = torch.tensor(tag_idx, device=flair.device)
        tag_list.append(tag)
    return tag_list