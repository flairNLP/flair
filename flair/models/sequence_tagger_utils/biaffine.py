import torch
import flair
from flair.data import Dictionary, DataPoint
from typing import List, Union
class Biaffine(torch.nn.Module):

    def __init__(self, ffnn_input_size: int, ffnn_output_size: int, ffnn_dropout: int, tag_dictionary_lenght: int, init_from_state_dict: bool):
        super(Biaffine, self).__init__()

        self.ffnn_start = torch.nn.Sequential(
            torch.nn.Linear(ffnn_input_size, ffnn_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout))

        self.ffnn_end = torch.nn.Sequential(
            torch.nn.Linear(ffnn_input_size, ffnn_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout))

        self.weight = torch.nn.Parameter(torch.Tensor(tag_dictionary_lenght, ffnn_output_size + 1, ffnn_output_size + 1))
        if not init_from_state_dict:
            torch.nn.init.zeros_(self.weight)

        self.to(flair.device)

    def forward(self, features: torch.tensor):

        embed_start = self.ffnn_start(features)
        embed_end = self.ffnn_start(features)

        x = torch.cat([embed_start, embed_start.new_ones(embed_start.shape[:-1]).unsqueeze(-1)], -1)
        y = torch.cat([embed_end, embed_end.new_ones(embed_end.shape[:-1]).unsqueeze(-1)], -1)

        x.unsqueeze_(1)
        y.unsqueeze_(1)

        output = x.matmul(self.weight).matmul(y.transpose(-1, -2))
        output.squeeze_(1)

        candidate = output.permute(0, 2, 3, 1).contiguous()

        return candidate



class BiaffineDecoder:

    def __init__(self, tag_dictionary: Dictionary):

        self.label_dictionary = tag_dictionary

    def decode(self, features, batch, is_flat_ner, return_probabilities_for_all_classes):
        all_tags = []
        candidates = []
        for sid,sent in enumerate(batch):
            for s in range(len(sent)):
                for e in range(s,len(sent)):
                    candidates.append((sid,s,e))

        top_spans = [[] for _ in range(len(batch))]
        for i, ner in enumerate(features.argmax(axis=1)):
            if ner > 0:
                sid, s,e = candidates[i]
                top_spans[sid].append((s, e, ner, features[i, ner]))

        top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans]
        sent_pred_mentions = [[] for _ in range(len(batch))]
        for sid, top_span in enumerate(top_spans):
            for ns, ne, tag, score in top_span:
                for ts, te, _, _ in sent_pred_mentions[sid]:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        #for both nested and flat ner no clash is allowed
                        break
                    if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                        #for flat ner nested mentions are not allowed
                        break
                else:
                    sent_pred_mentions[sid].append((ns, ne, tag, score))

        predictions = []
        for spr in sent_pred_mentions:
            spans = []
            for ns, ne, ner, score in spr:
                if ns < ne:
                    spans.append(([ns+1, ne+1], score.item(), self.label_dictionary.get_item_for_index(ner.item())))
                else:
                    spans.append(([ns+1], score.item(), self.label_dictionary.get_item_for_index(ner.item())))
            predictions.append(spans)

        return predictions, all_tags

    def get_labels4biaffine(self, sentences: Union[List[DataPoint], DataPoint]):

        gold_labels = []
        for sentence in sentences:
            ner = {(label.span[0].idx-1, label.span[0].idx-1):label.value for label in sentence.get_labels("ner")}
            for s in range(0, len(sentence)):
                for e in range(s,len(sentence)):
                    gold_labels.append([ner.get((s,e),"0")])

        return gold_labels

    def get_useful4biaffine(self, lengths, candidate):

        # generate mask
        lengths = lengths.values
        longest_token_sequence_in_batch = max(lengths)
        mask = [[1] * lengths[i] + [0] * (longest_token_sequence_in_batch - lengths[i]) for i in range(len(lengths))]
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
        mask = torch.triu(mask)
        mask = mask.reshape(-1)

        tmp_candidate = candidate.reshape(-1, candidate.shape[-1])
        indices = mask.nonzero(as_tuple=False).squeeze(-1).long().to(flair.device)
        scores = tmp_candidate.index_select(0, indices)

        return scores