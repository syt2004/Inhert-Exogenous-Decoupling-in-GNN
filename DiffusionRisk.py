import torch.nn as nn
import torch


class DiffusionRisk(nn.Module):
    def __init__(self,riskFactor):
        super().__init__()
        self.riskFactor =riskFactor
        self.ac=nn.Sigmoid()
    def forward(self, dif_data):
        hid=dif_data*self.riskFactor
        hid=self.ac(hid)
        return hid

# class DiffusionRisk(nn.Module):
#     def __init__(self,batch_size,seg_length,node,feature):
#         super().__init__()
#         self.riskFactor =nn.Parameter(torch.empty(batch_size,seg_length,node,feature))
#         self.ac=nn.Sigmoid()
#     def forward(self, dif_data):
#         hid=dif_data*self.riskFactor
#         hid=self.ac(hid)
#         return hid
