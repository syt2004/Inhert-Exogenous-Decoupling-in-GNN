import torch.nn as nn
from GRU import GRU
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalBlock, self).__init__()
        self.gru = GRU(in_channels=in_channels, out_channels=out_channels, K=1, normalization='sym')
        self.attention = nn.MultiheadAttention(out_channels, num_heads=4)

    def forward(self, x, edge_index):
        x = x.permute(2, 1, 0, 3).contiguous()  # [num_nodes, seq_length, batch_size, features]
        num_nodes, seq_length, batch_size, features = x.shape
        x = x.view(num_nodes, seq_length * batch_size, features)
        x = self.gru(x, edge_index)
        x = x.view(num_nodes, seq_length, batch_size, -1)
        x = x.permute(2, 1, 0, 3).contiguous()  # [batch_size, seq_length, num_nodes, features]
        x = x.view(batch_size * seq_length, num_nodes, -1)
        x, _ = self.attention(x, x, x)
        x = x.view(batch_size, seq_length, num_nodes, -1)
        return x