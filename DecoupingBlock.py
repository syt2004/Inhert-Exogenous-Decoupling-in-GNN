from EstimationGate import EstimationGate
import torch.nn as nn


class DecouplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, node_emb_dim, time_emb_dim, hidden_dim):
        super(DecouplingBlock, self).__init__()
        self.estimation_gate = EstimationGate(node_emb_dim, time_emb_dim, hidden_dim)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat):
        x=self.linear(x)
        gated_data=self.estimation_gate(node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat, x)
        return x,gated_data