import torch


def cdg(predicted_scores, true_scores, topk, ver='traditional'):
    max_topk = max(topk)
    n, k = true_scores.shape
    log2_positions_plus_1 = torch.log2((torch.arange(max_topk) + 2).float()).unsqueeze(0)
    if predicted_scores is not None:
        sorting_indices = torch.sort(predicted_scores, descending=True, dim=1)[1]
        row_indices = torch.arange(n).unsqueeze(1) * torch.ones(n,max_topk).long()
        true_sorted = true_scores[row_indices, sorting_indices[:,:max_topk]]
    else:
        true_sorted = true_scores[:,:max_topk]
    if ver == 'alternative':
        true_sorted = torch.pow(2, true_sorted) - 1
    DCG_vals = true_sorted / log2_positions_plus_1
    DCGs = torch.stack([DCG_vals[:,:rank].sum(dim=1) for rank in topk], dim=1)
    return DCGs

def ndcg(predicted_scores, true_scores, topk, ver='traditional'):
    cdgs = cdg(predicted_scores, true_scores, topk, ver)
    icdgs = cdg(None, torch.sort(true_scores, descending=True, dim=1)[0], topk, ver)
    return cdgs / icdgs

def mncdg(predicted_scores, true_scores, topk=[1,20,50,80,120,240], ver='traditional', weighted=True):
    if weighted:
        # weight the scores by the proportion
        true_scores_proportion = true_scores.sum(dim=1, keepdim=True)
        true_scores_proportion /= true_scores_proportion.sum()
    else:
        n = true_scores.shape[0]
        true_scores_proportion = torch.ones((n,1)) / n
    return (true_scores_proportion * ndcg(predicted_scores, true_scores, topk, ver)).sum(dim=0)
