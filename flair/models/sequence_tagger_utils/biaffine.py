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
        batch_size = features.size()[0]
        longest_token_sequence_in_batch = features.size()[1]
        output_size = self.bilinear_map.size()[1]
        embed_start = self.ffnn_start(features)
        embed_end = self.ffnn_start(features)
        #  [batch, longest_token_sequence_in_batch, ffnn_output_size + 1]
        vector_set_1 = torch.cat([embed_start, embed_start.new_ones(embed_start.shape[:-1]).unsqueeze(-1)], -1)
        vector_set_2 = torch.cat([embed_end, embed_end.new_ones(embed_end.shape[:-1]).unsqueeze(-1)], -1)

        vector_set_1_size = vector_set_1.size()[-1]
        vector_set_2_size = vector_set_2.size()[-1]

        # The matrix operations and reshapings for bilinear mapping.
        # b: batch size (batch of buckets)
        # v1, v2: values (size of vectors)
        # n: tokens (size of bucket)
        # r: labels (output size), e.g. 1 if unlabeled or number of edge labels.
        # # [b, n, v1] -> [b*n, v1]
        vector_set_1 = vector_set_1.view(-1, vector_set_1_size,)

        # [v1, r, v2] -> [v1, r*v2]
        bilinear_mapping = self.bilinear_map.view(vector_set_1_size, -1)

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = vector_set_1.matmul(bilinear_mapping)

        # [b*n, r*v2] -> [b, n*r, v2]
        bilinear_mapping = bilinear_mapping.view(batch_size, batch_size*longest_token_sequence_in_batch, vector_set_2_size)

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        vector_set_2_size = torch.conj(vector_set_2).transpose(2, 1)
        output = bilinear_mapping.bmm(vector_set_2_size)
        # [b, n*r, n] -> [b, n, r, n]
        output = output.view(batch_size, longest_token_sequence_in_batch, output_size, longest_token_sequence_in_batch)
        # [b, n, r, n] -> [b, n, n, r]
        candidate = output.permute(0,1,3,2).contiguous()

        return candidate



class BiaffineDecoder:

    def __init__(self, tag_dictionary: Dictionary):

        self.label_dictionary = tag_dictionary

    def decode(self, features, batch, is_flat_ner, return_probabilities_for_all_classes):
        # TODO all_tags, return_probabilities_for_all_classes
        all_tags = []
        candidates = []
        outside = self.label_dictionary.get_idx_for_item('O')

        # get prediction matrix
        for sid,sent in enumerate(batch):
            for s in range(len(sent)):
                for e in range(s,len(sent)):
                    candidates.append((sid,s,e))
        # Find all possible positions in prediction matrix
        top_spans = [[] for _ in range(len(batch))]
        for i, ner in enumerate(features.argmax(axis=1)):
            if ner != 0 and ner != outside:
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
                    spans.append(([ns+1, ne+1], score.item(), self.label_dictionary.get_item_for_index(ner.item())))
                else:
                    spans.append(([ns+1], score.item(), self.label_dictionary.get_item_for_index(ner.item())))
            predictions.append(spans)

        return predictions, all_tags

    def get_labels4biaffine(self, sentences: Union[List[DataPoint], DataPoint]):

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
            ner = {(label.span[0].idx-1, label.span[-1].idx-1):label.value for label in sentence.get_labels("ner")}
            for s in range(0, len(sentence)):
                for e in range(s,len(sentence)):
                    gold_labels.append([ner.get((s,e),"O")])

        return gold_labels

    def get_useful4biaffine(self, lengths, candidate):

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

    # def get_label_weight(self, sentences: Union[List[DataPoint], DataPoint]):
    #     all_label = self.get_labels4biaffine(sentences)
