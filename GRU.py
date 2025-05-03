import torch
import torch.nn as nn




class GRU(nn.Module):
    def __init__(self, in_channels, out_channels, K, normalization='sym'):
        super(GRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.update_gate = nn.Linear(in_channels + out_channels, out_channels)
        self.reset_gate = nn.Linear(in_channels + out_channels, out_channels)
        self.out_gate = nn.Linear(in_channels + out_channels, out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.size(0), X.size(1), self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.update_gate(torch.cat([X, H], dim=-1))
        return torch.sigmoid(Z)

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.reset_gate(torch.cat([X, H], dim=-1))
        return torch.sigmoid(R)

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        RH = R * H
        H_tilde = self.out_gate(torch.cat([X, RH], dim=-1))
        return torch.tanh(H_tilde)

    def _calculate_hidden_state(self, Z, H, H_tilde):
        return Z * H + (1 - Z) * H_tilde

    def forward(self, X, edge_index, edge_weight=None, H=None, lambda_max=None):
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H