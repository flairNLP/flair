import torch

# Source: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L23


def dot_product(a: torch.Tensor, b: torch.Tensor, normalize=False):
    """
    Computes dot product for pairs of vectors.
    :param normalize: Vectors are normalized (leads to cosine similarity)
    :return: Matrix with res[i][j]  = dot_product(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    if normalize:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a, b.transpose(0, 1))


class CosineDistance(torch.nn.Module):
    def forward(self, a, b):
        return -dot_product(a, b, normalize=True)


class LogitCosineDistance(torch.nn.Module):
    def forward(self, a, b):
        return torch.logit(0.5 - 0.5 * dot_product(a, b, normalize=True))


class NegativeScaledDotProduct(torch.nn.Module):
    def forward(self, a, b):
        sqrt_d = torch.sqrt(torch.tensor(a.size(-1)))
        return -dot_product(a, b, normalize=False) / sqrt_d
