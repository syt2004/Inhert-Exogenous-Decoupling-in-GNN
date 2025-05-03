import torch
import torch.nn as nn

from cal_adj import remove_nan_inf


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        degree  = remove_nan_inf(1 / degree)
        degree  = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]

