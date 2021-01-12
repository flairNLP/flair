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

    padded_tags_list = pad_sequence(tag_list, batch_first = True)

    # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)
    crf_tags = list(map(
                lambda s: [tag_dictionary.get_idx_for_item('<START>') * len(tag_dictionary) + s[0]] + [s[i - 1] * len(tag_dictionary) + s[i]
                for i in range(1, len(s))], padded_tags_list))

    tags_tensor = torch.LongTensor(crf_tags)

    return tags_tensor