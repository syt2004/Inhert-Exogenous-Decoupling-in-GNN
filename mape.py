import argparse
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from dataloader import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def load_data(data_dir, batch_size):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(data_dir, f"{category}.npz"))
        xs, ys = cat_data["x"], cat_data["y"]
        data[category] = DataLoader(xs, ys, batch_size)
    return data

def load_adj(pkl_path):
    with open(pkl_path, 'rb') as f:
        adj_mx = pickle.load(f)
    return adj_mx

class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat, history_data):
        batch_size, seq_length, _, _ = time_in_month_feat.shape
        estimation_gate_feat = torch.cat([time_in_month_feat, time_in_year_feat,
                                          node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
                                          node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data


class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u

class DecouplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, node_emb_dim, time_emb_dim, hidden_dim):
        super(DecouplingBlock, self).__init__()
        self.estimation_gate = EstimationGate(node_emb_dim, time_emb_dim, hidden_dim)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat):
        x = self.linear(x)
        gated_data = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat, x)
        return x, gated_data

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

class LowPassFilter(nn.Module):
    def __init__(self):
        super(LowPassFilter, self).__init__()

    def forward(self, x):
        x_freq = torch.fft.fft(x, dim=-1)
        freqs = torch.fft.fftfreq(x.size(-1)).to(x.device)
        mask = (torch.abs(freqs) < 0.5).to(x.device)
        x_freq = x_freq * mask
        x_time = torch.fft.ifft(x_freq, dim=-1).real
        return x_time

class DecoupledModel(nn.Module):
    def __init__(self, node_features, out_channels, node_emb_dim, time_emb_dim, hidden_dim):
        super(DecoupledModel, self).__init__()
        self.decoupling_block = DecouplingBlock(node_features, out_channels, node_emb_dim, time_emb_dim, hidden_dim)
        self.residual = ResidualDecomp([-1, -1, -1, hidden_dim])
        self.temporal_block = TemporalBlock(out_channels, out_channels)
        self.low_pass_filter = LowPassFilter()
        self.linear = nn.Linear(out_channels, 1)
        self.dif_hid = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat):
        x, diffusion_signal = self.decoupling_block(x, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat)
        # 扩散信号用频域处理
        diffusion_signal = self.low_pass_filter(diffusion_signal)
        diffusion_forcast = self.linear(diffusion_signal)

        diffusion_hid = self.dif_hid(diffusion_signal)
        history_data = x[:, -diffusion_hid.shape[1]:, :, :]
        diffusion_res = self.residual(history_data, diffusion_hid)

        inherent_signal = x - diffusion_signal + diffusion_res
        # 固有信号用时域处理
        inherent_signal = self.temporal_block(inherent_signal, edge_index)
        inherent_forcast = self.linear(inherent_signal)
        x = inherent_forcast + diffusion_forcast
        x = x[:, -1, :, :]
        x = x.unsqueeze(1)
        return x


def mape(pred, true):
    mask = true != 0
    return torch.mean(torch.abs((true[mask] - pred[mask]) / true[mask])) * 100


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Chickenpox', help='Dataset name.')
args = parser.parse_args()
config_path = "configs/" + args.dataset + ".yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
data_dir = config['data_args']['data_dir']
dataset_name = config['data_args']['data_dir'].split("/")[-1]

device = torch.device(config['start_up']['device'])
save_path = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
save_path_resume = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
load_pkl = config['start_up']['load_pkl']
model_name = config['start_up']['model_name']

batch_size = config['model_args']['batch_size']
dataloader = load_data(data_dir, batch_size)
pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))

adj_mx = load_adj(config['data_args']['adj_data_path'])

edge_index = torch.tensor(np.nonzero(adj_mx), dtype=torch.long, device=device)

num_nodes = adj_mx.shape[0]
loop_index = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)

edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)

node_dim = 32
time_emb_dim = 4
hidden_dim = 64
out_channels = 64
node_emb_u = nn.Parameter(torch.empty(num_nodes, node_dim)).to(device)
node_emb_d = nn.Parameter(torch.empty(num_nodes, node_dim)).to(device)
T_i_M_emb = nn.Parameter(torch.empty(30, time_emb_dim)).to(device)  # 修改为每月时间特征
T_i_Y_emb = nn.Parameter(torch.empty(12, time_emb_dim)).to(device)  # 修改为每年时间特征
nn.init.xavier_uniform_(node_emb_u)
nn.init.xavier_uniform_(node_emb_d)
nn.init.xavier_uniform_(T_i_M_emb)
nn.init.xavier_uniform_(T_i_Y_emb)

def prepare_data(x, node_emb_u, node_emb_d, T_i_M_emb, T_i_Y_emb):
    if x.ndim == 3:
        x = x.unsqueeze(-1)

    time_in_month_feat = T_i_M_emb[(x[:, :, :, 1] * 30).clamp(0, 29).type(torch.LongTensor).to(device)]  # 修改为每月时间特征并限制范围
    time_in_year_feat = T_i_Y_emb[(x[:, :, :, 2] * 12).clamp(0, 11).type(torch.LongTensor).to(device)]  # 修改为每年时间特征并限制范围
    return x, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat


model = DecoupledModel(node_features=3, out_channels=out_channels, node_emb_dim=node_dim, time_emb_dim=time_emb_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1.0e-5)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(50):
    total_loss = 0
    total_mape = 0
    for x, y in dataloader['train'].get_iterator():
        x, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat = prepare_data(torch.tensor(x, dtype=torch.float).to(device), node_emb_u, node_emb_d, T_i_M_emb, T_i_Y_emb)
        y = torch.tensor(y[..., 0:1], dtype=torch.float).to(device)  # 取出感染数特征
        optimizer.zero_grad()
        out = model(x, edge_index_with_loops, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mape += mape(out, y).item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader["train"]):.4f}, MAPE: {total_mape / len(dataloader["train"]):.4f}')

model.eval()
test_mse = 0
test_mae = 0
test_mape = 0
with torch.no_grad():
    for x, y in dataloader['test'].get_iterator():
        x, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat = prepare_data(torch.tensor(x, dtype=torch.float).to(device), node_emb_u, node_emb_d, T_i_M_emb, T_i_Y_emb)
        y = torch.tensor(y[..., 0:1], dtype=torch.float).to(device)  # 取出感染数特征
        out = model(x, edge_index_with_loops, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat)
        mse = loss_fn(out, y).item()
        mae = F.l1_loss(out, y).item()
        mape_value = mape(out, y).item()
        test_mse += mse
        test_mae += mae
        test_mape += mape_value
print(f'Test MSE: {test_mse / len(dataloader["test"]):.4f}, Test MAE: {test_mae / len(dataloader["test"]):.4f}, Test MAPE: {test_mape / len(dataloader["test"]):.4f}')

model.eval()
x, y = next(dataloader['test'].get_iterator())
x, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat = prepare_data(torch.tensor(x, dtype=torch.float).to(device), node_emb_u, node_emb_d, T_i_M_emb, T_i_Y_emb)
y = torch.tensor(y[..., 0:1], dtype=torch.float).to(device)  # 取出感染数特征
out = model(x, edge_index_with_loops, node_emb_u, node_emb_d, time_in_month_feat, time_in_year_feat)

plt.figure(figsize=(10, 5))
plt.plot(y.cpu().numpy().flatten(), label='True')
plt.plot(out.cpu().detach().numpy().flatten(), label='Predicted')
plt.legend()
plt.show()
