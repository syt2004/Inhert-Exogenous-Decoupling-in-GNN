import torch
import torch.nn as nn
from  Dynamic_graph import DynamicGraphConstructor
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


class FrequencyBlock(nn.Module):
    def __init__(self,k_s, k_t,num_hidden,node_hidden,seq_length,dropout,time_emb_dim,adjs):
        super(FrequencyBlock, self).__init__()
        self.low_pass = LowPassFilter()
        self.dynamic_graph=DynamicGraphConstructor(k_s, k_t,num_hidden,node_hidden,seq_length,dropout,time_emb_dim,adjs)

    def forward(self,history_data,node_embedding_d,node_embedding_u,time_in_day_feat,day_in_week_feat):
        x=self.low_pass(history_data)
        x_graphed=self.dynamic_graph(x,node_embedding_d,node_embedding_u,time_in_day_feat,day_in_week_feat)
        return x_graphed