import torch
import flair

class Biaffine(torch.nn.Module):

    def __init__(self, embedding_dim: int, ffnn_size: int, ffnn_dropout: int, tag_dictionary_lenght: int, init_from_state_dict: bool):
        super(Biaffine, self).__init__()

        self.ffnn_start = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffnn_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout))

        self.ffnn_end = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffnn_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(ffnn_dropout))

        self.weight = torch.nn.Parameter(torch.Tensor(tag_dictionary_lenght, ffnn_size + 1, ffnn_size + 1))
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

    def decode(self, sentences, span_scores, is_flat_ner):
        candidates = []
        for sid,sent in enumerate(sentences):
            for s in range(len(sent)):
                for e in range(s,len(sent)):
                    candidates.append((sid,s,e))

        top_spans = [[] for _ in range(len(sentences))]
        for i, ner in enumerate(span_scores.argmax(axis=1)):
            if ner > 0:
                sid, s,e = candidates[i]
                top_spans[sid].append((s,e,ner,span_scores[i,ner]))

        top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans]
        sent_pred_mentions = [[] for _ in range(len(sentences))]
        for sid, top_span in enumerate(top_spans):
            for ns,ne,t,_ in top_span:
                for ts,te,_ in sent_pred_mentions[sid]:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        #for both nested and flat ner no clash is allowed
                        break
                    if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                        #for flat ner nested mentions are not allowed
                        break
                else:
                    sent_pred_mentions[sid].append((ns,ne,t))
        pred_mentions = set((sid,s,e,t) for sid, spr in enumerate(sent_pred_mentions) for s,e,t in spr)

        return pred_mentions