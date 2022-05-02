import torch
import flair
from flair.data import Dictionary, DataPoint
from typing import List, Union


class Biaffine(torch.nn.Module):

    def __init__(self, ffnn_input_size: int, ffnn_output_size: int, ffnn_dropout: int, output_size: int, init_from_state_dict: bool):
        super(Biaffine, self).__init__()

        self.ffnn_start = torch.nn.Sequential(
            torch.nn.Linear(ffnn_input_size, ffnn_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout),
            torch.nn.Linear(ffnn_output_size, ffnn_output_size),
            torch.nn.Dropout(ffnn_dropout))

        self.ffnn_end = torch.nn.Sequential(
            torch.nn.Linear(ffnn_input_size, ffnn_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout),
            torch.nn.Linear(ffnn_output_size, ffnn_output_size),
            torch.nn.Dropout(ffnn_dropout))

        self.bilinear_map = torch.nn.Parameter(torch.Tensor(ffnn_output_size + 1, output_size, ffnn_output_size + 1))
        if not init_from_state_dict:
            torch.nn.init.zeros_(self.bilinear_map)

        self.to(flair.device)

    def forward(self, features: torch.tensor):
        # input: [batch, longest_token_sequence_in_batch, ffnn_input_size]
        embed_start = self.ffnn_start(features)
        embed_end = self.ffnn_end(features)
        #  [batch, longest_token_sequence_in_batch, ffnn_output_size + 1]
        embed_start = torch.cat([embed_start, embed_start.new_ones(embed_start.shape[:-1]).unsqueeze(-1)], -1)
        embed_end = torch.cat([embed_end, embed_end.new_ones(embed_end.shape[:-1]).unsqueeze(-1)], -1)

        output = torch.einsum('bxi,oij,byj->boxy', embed_start, self.bilinear_map, embed_end)
        # remove dim 1 if n_out == 1
        output = output.squeeze(1)
        candidate = output.permute(0,2,3,1).contiguous()

        return candidate



class BiaffineDecoder:

    def __init__(self, tag_dictionary: Dictionary):

        self.label_dictionary = tag_dictionary

    def decode(self, features, batch, is_flat_ner):

        candidates = []
        outside = self.label_dictionary.get_idx_for_item('O')

        # get prediction matrix
        for sid,sent in enumerate(batch):
            for s in range(len(sent)):
                for e in range(s,len(sent)):
                    candidates.append((sid,s,e))
        # Find all possible positions in prediction matrix
        top_spans = [[] for _ in range(len(batch))]
        for i, ner in enumerate(features.argmax(axis=-1)):
            if ner != outside:
                sid, s,e = candidates[i]
                top_spans[sid].append((s, e, ner, features[i, ner]))
        # Sort by predicted score
        top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans]
        sent_pred_mentions = [[] for _ in range(len(batch))]
        # handle conflict
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

        # process the predicted results into the format required by predict
        predictions = []
        for spr in sent_pred_mentions:
            spans = []
            for ns, ne, ner, score in spr:
                if ns < ne:
                    spans.append(([ns, ne], score.item(),  self.label_dictionary.get_item_for_index(ner.item())))
                else:
                    spans.append(([ns, ns], score.item(),  self.label_dictionary.get_item_for_index(ner.item())))
            predictions.append(spans)

        return predictions

    def get_labels(self, sentences: Union[List[DataPoint], DataPoint]):

        """
        :param sentences: list of Sentence
        prediction matrix example:
        Sentence: The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb .
        'ner': [ORG [European Commission (2,3)] (1.0), MISC [German (10)] (1.0), MISC [British (16)] (1.0)]}
                   The  European  Commission  said  on  Thursday  it  disagreed  with  German  advice  to  consumers  to  shun  British  lamb  .
        The         O      O          O        O      O       O    O      O        O     O        O     O      O       O    O      O       O   O
        European           O         ORG       O      O       O    O      O        O     O        O     O      O       O    O      O       O   O
        Commission                    O        O      O       O    O      O        O     O        O     O      O       O    O      O       O   O
        said                                   O      O       O    O      O        O     O        O     O      O       O    O      O       O   O
        on                                            O       O    O      O        O     O        O     O      O       O    O      O       O   O
        Thursday                                              O    O      O        O     O        O     O      O       O    O      O       O   O
        it                                                         O      O        O     O        O     O      O       O    O      O       O   O
        disagreed                                                         O        O     O        O     O      O       O    O      O       O   O
        with                                                                       O     O        O     O      O       O    O      O       O   O
        German                                                                          MISC      O     O      O       O    O      O       O   O
        advice                                                                                    O     O      O       O    O      O       O   O
        to                                                                                              O      O       O    O      O       O   O
        consumers                                                                                              O       O    O      O       O   O
        to                                                                                                             O    O      O       O   O
        shun                                                                                                                O      O       O   O
        British                                                                                                                   MISC     O   O
        lamb                                                                                                                               O   O
        .                                                                                                                                      O
        :return: 1-D  gold labels
        """

        gold_labels = []
        for sentence in sentences:
            ner = {(label.data_point.tokens[0].idx-1, label.data_point.tokens[-1].idx-1):label.value for label in sentences[0].get_labels("ner")}
            for s in range(0, len(sentence)):
                for e in range(s,len(sentence)):
                    gold_labels.append([ner.get((s,e),"0")])

        return gold_labels

    def get_flat_scores(self, lengths, candidate):

        # extracting useful predictions from the prediction matrix, and store all predictions in a one-dimensional tensor, corresponding to the labels
        # generate mask
        lengths = lengths.values
        longest_token_sequence_in_batch = max(lengths)
        # [batch, longest_token_sequence_in_batch]
        mask = [[1] * lengths[i] + [0] * (longest_token_sequence_in_batch - lengths[i]) for i in range(len(lengths))]
        mask = torch.tensor(mask)
        # [batch, longest_token_sequence_in_batch, longest_token_sequence_in_batch]
        mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
        mask = torch.triu(mask)
        mask = mask.reshape(-1)

        tmp_candidate = candidate.reshape(-1, candidate.shape[-1])
        indices = mask.nonzero(as_tuple=False).squeeze(-1).long().to(flair.device)
        scores = tmp_candidate.index_select(0, indices)

        return scores

