import numpy as np
import torch.nn as nn
import torch
from Distance import DistanceFunction
from Mask import Mask
from Normalize import Normalizer
from MultiOrder import MultiOrder
class DynamicGraphConstructor(nn.Module):
    def __init__(self, k_s, k_t,num_hidden,node_hidden,seq_length,dropout,time_emb_dim,adjs):
        super().__init__()
        # model args
        self.k_s = k_s # spatial order
        self.k_t = k_t # temporal kernel size
        # hidden dimension of
        self.hidden_dim = num_hidden
        # trainable node embedding dimension
        self.node_dim = node_hidden

        self.distance_function = DistanceFunction(num_hidden,node_hidden,seq_length,dropout,time_emb_dim)
        self.mask = Mask(adjs)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self,history_data,node_embedding_d,node_embedding_u,time_in_day_feat,day_in_week_feat):
        """Dynamic graph learning module.

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            list: dynamic graphs
        """

        X = history_data
        E_d = node_embedding_d
        E_u = node_embedding_u
        T_D = time_in_day_feat
        D_W = day_in_week_feat
        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # multi order
        mul_mx = self.multi_order(dist_mx)
        #print(mul_mx.shape)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)

        return dynamic_graphs