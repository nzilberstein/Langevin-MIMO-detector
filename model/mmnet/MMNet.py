import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
# from numpy import linalg as LA
import scipy.linalg as LA
import pandas as pd
from MMNet_base import *

class MMNet(nn.Module):

    def __init__(self, num_layers, NT, NR, constellation, device='cuda'):
        super(MMNet, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.NT = NT
        self.NR = NR
        self.constellation = constellation
        self.MMNetbase = nn.ModuleList([MMNet_base(self.NT, self.NR, self.constellation, self.device) for i in range(self.num_layers)])

    def forward(self, H, y, noise_sigma, batch_size):
        x_size = H.shape[-1]

        x_prev = torch.zeros(batch_size, x_size, dtype=torch.double).to(device=self.device)
        x_list=[x_prev]

        for index, MMNetbase in enumerate(self.MMNetbase):
            xout = MMNetbase.forward(H, y, x_list[-1].double(), noise_sigma, batch_size)
            x_list.append(xout.double())
        return (x_list[1:])