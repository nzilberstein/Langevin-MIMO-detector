"""
MLP network.py MLP for unfolded Langevin detector
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MLPNetwork(nn.Module):
    def __init__(self, NT, NR, device):
        super(MLPNetwork, self).__init__()
        self.input_shape = 4 * NT + 1
        self.output_shape = 2 * NT
        self.device = device
        self.hidden1 = 400
        self.hidden2 = 350
        self.hidden3 = 100

        self.layer1 = nn.Linear(self.input_shape , self.hidden1).to(device=self.device).double()
        self.layer2 = nn.Linear(self.hidden1 , self.hidden2).to(device=self.device).double()
        self.layer3 = nn.Linear(self.hidden2 , self.hidden3).to(device=self.device).double()
        self.layer4 = nn.Linear(self.hidden3, self.output_shape).to(device=self.device).double()

        self.layer1.weight = torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2.weight = torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3.weight = torch.nn.init.xavier_uniform_(self.layer3.weight)
        self.layer4.weight = torch.nn.init.xavier_uniform_(self.layer4.weight)


    def process_forward(self, H):

        output1 = self.layer1(H.double())
        norm_output1 = F.elu(output1, alpha = 3)
        output2 = self.layer2(norm_output1)
        norm_output2 = F.elu(output2, alpha = 3)
        output3 = self.layer3(norm_output2)
        norm_output3 = F.elu(output3, alpha = 3)
        
        paramsMainNet = self.layer4(norm_output3)
        
        return paramsMainNet

    def forward(self, H):
        return self.process_forward(H)