import torch
from torch.nn.utils.rnn import pad_sequence

import flair
from flair.data import List

def get_tags_tensor(sentences, tag_dictionary, tag_type):
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

    padded_tag_tensor = pad_sequence(tag_list, batch_first=True)

    return padded_tag_tensor