import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from oampnet_base import oampnet_base

class oampnet(nn.Module):

    def __init__(self, num_layers, constel, real_QAM_const, imag_QAM_const, device='cpu'):
        super(oampnet, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.oampbases = nn.ModuleList([oampnet_base(constel, real_QAM_const, imag_QAM_const, device) for i in range(self.num_layers)])

    def forward(self, H, y, noise_sigma):
        batch_size = H.shape[0]
        x_size = H.shape[-1]

        x_prev = torch.zeros(batch_size, x_size).to(device=self.device)
        x_list=[x_prev]
      
        for index,oampbase in enumerate(self.oampbases):
            xout = oampbase.forward(H, y, x_list[-1], noise_sigma)
            x_list.append(xout)
        return (x_list[1:])