import pandas as pd

from load_data import load_data
from load_adj import load_adj
from ResidualDecomp import ResidualDecomp
from DecoupingBlock import DecouplingBlock
from TemporalBlock import TemporalBlock
from prepare_data_week_month import prepare_data
from FrequencyBlock import FrequencyBlock

import argparse
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

# from dataloader import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




class IEDGNN(nn.Module):
    def __init__(self, node_features, out_channels=64, node_emb_dim=64, time_emb_dim=4, hidden_dim=64, lr=0.002,
             k_s=4, k_t=1, num_hidden=32, node_hidden=32, seq_length=8, dropout=491/50000, adjs=1):
        super(IEDGNN, self).__init__()  # seq_length;node_hidden;node_features不能优化
        self.decoupling_block = DecouplingBlock(node_features, out_channels, node_emb_dim, time_emb_dim, hidden_dim)
        self.residual = ResidualDecomp([-1, -1, -1, hidden_dim])
        self.temporal_block = TemporalBlock(out_channels, out_channels)
        self.frequency_block = FrequencyBlock(k_s, k_t, num_hidden, node_hidden, seq_length, dropout, time_emb_dim,
                                              adjs)
        self.dif_linear = nn.Linear(11, 1)  # in_features必须是num_region
        self.inh_linear = nn.Linear(out_channels, 1)
        self.dif_hid = nn.Linear(11, hidden_dim)  # in_features必须是20
        self.lr = lr

    def forward(self, x, edge_index, node_embedding_u, node_embedding_d, time_in_month_feat, time_in_year_feat):
        x, diffusion_signal_o = self.decoupling_block(x, node_embedding_u, node_embedding_d,
                                                      time_in_month_feat, time_in_year_feat)
        # 扩散信号用频域处理
        diffusion_signal = self.frequency_block(diffusion_signal_o, node_embedding_d, node_embedding_u,
                                                time_in_month_feat, time_in_year_feat)
        diffusion_signal = torch.stack(diffusion_signal, dim=1).float()

        diffusion_forcast = self.dif_linear(diffusion_signal)

        diffusion_hid = self.dif_hid(diffusion_signal)
        history_data = x[:, -diffusion_hid.shape[1]:, :, :]
        diffusion_res = self.residual(history_data, diffusion_hid)

        inherent_signal = x - diffusion_signal_o + diffusion_res
        # 固有信号用时域处理
        inherent_signal = self.temporal_block(inherent_signal, edge_index)
        inherent_forcast = self.inh_linear(inherent_signal)
        x = inherent_forcast + diffusion_forcast
        x = x[:, -1, :, :]
        x = x.unsqueeze(1)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='JERCom2012', help='Dataset name.')
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
    dataloader = load_data(data_dir, batch_size,10)
    pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))

    adj_mx = load_adj(config['data_args']['adj_data_path'])

    edge_index = torch.tensor(np.nonzero(adj_mx), dtype=torch.long, device=device)

    num_nodes = adj_mx.shape[0]
    loop_index = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)

    edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)

    node_dim = 32
    time_emb_dim = 4
    hidden_dim = 86 * 4  # ok for nega, but must be equal to out_channels
    out_channels = 86 * 4  # ok for nega, but must can be divide by heads in temporal block which is equal to 4
    device = torch.device("cuda:0")

    adj_mx = np.array([[float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else np.nan for x in row] for row in adj_mx])

    # 将 NaN 值替换成 0 或其他默认值
    adj_mx = np.nan_to_num(adj_mx)

    adj_mx = np.array(adj_mx, dtype=np.float32)

    adjs = [torch.tensor(i).to(device) for i in adj_mx ]
    node_emb_u = nn.Parameter(torch.empty(num_nodes, node_dim)).to(device)
    node_emb_d = nn.Parameter(torch.empty(num_nodes, node_dim)).to(device)
    T_i_W_emb = nn.Parameter(torch.empty(7, time_emb_dim)).to(device)  # 修改为每周时间特征
    T_i_M_emb = nn.Parameter(torch.empty(30, time_emb_dim)).to(device)  # 修改为每月时间特征
    nn.init.xavier_uniform_(node_emb_u)
    nn.init.xavier_uniform_(node_emb_d)
    nn.init.xavier_uniform_(T_i_W_emb)
    nn.init.xavier_uniform_(T_i_M_emb)

    '''
    # num_hidden is ok for nega
    hidden_dim ok for nega, but must be equal to out_channels
    out_channels ok for nega, but must can be divide by heads in temporal block which is equal to 4
    '''

    model = IEDGNN(node_features=3, out_channels=out_channels, node_emb_dim=node_dim,
                       time_emb_dim=time_emb_dim, num_hidden=1,hidden_dim=hidden_dim, lr=3000 / 10000,
                       adjs=adjs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=201/500000, weight_decay=1.0e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        for x, y in dataloader['train'].get_iterator():
            x, node_emb_u, node_emb_d, time_in_week_feat, time_in_month_feat = prepare_data(
                torch.tensor(x, dtype=torch.float).to(device), node_emb_u, node_emb_d, T_i_W_emb, T_i_M_emb)
            y = torch.tensor(y[:, 0, :, 0:1], dtype=torch.float).unsqueeze(1).to(device)  # 取出感染数特征
            optimizer.zero_grad()
            out = model(x, edge_index_with_loops, node_emb_u, node_emb_d, time_in_week_feat,
                             time_in_month_feat)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader["train"]):.4f}')

    model.eval()
    test_mse = 0
    future = 1
    cnt = 0
    with torch.no_grad():
        for x, y in dataloader['test'].get_iterator():
            cnt += 1
            x, node_emb_u, node_emb_d, time_in_week_feat, time_in_month_feat = prepare_data(
                torch.tensor(x, dtype=torch.float).to(device), node_emb_u, node_emb_d, T_i_W_emb, T_i_M_emb)
            y_o = torch.tensor(y, dtype=torch.float).to(device)
            y = torch.tensor(y[..., 0:1], dtype=torch.float).to(device)  # 取出感染数特征
            for i in range(10):
                x = torch.cat((x.to(device), y_o[:, i, :, :].unsqueeze(1).to(device)), dim=1)
                out = model(x[:, -8:, :, :], edge_index_with_loops, node_emb_u, node_emb_d, time_in_week_feat,
                            time_in_month_feat)
                if i == 0:
                    future = out
                else:
                    future = torch.cat((future, out), dim=1)
            mse = loss_fn(future.to(device), y).item()
            test_mse += mse


        result = test_mse / len(dataloader["test"])
        print('Test MSE: ', result/10)


