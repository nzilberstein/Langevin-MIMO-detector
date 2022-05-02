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
from utils import *

class MMNet_base(nn.Module):

    def __init__(self, NT, NR, constellation, device):
        super(MMNet_base, self).__init__()

        self.device = device
        self.NT = 2*NT
        self.NR = 2*NR
        self.lgst_constel = constellation
        self.M = int(self.lgst_constel.shape[0])
        
        # self.Wr = nn.Parameter(torch.normal(0, 0.01, size=(1, int(self.NT/2), int(self.NR/2))))
        # self.Wi = nn.Parameter(torch.normal(0, 0.01, size=(1, int(self.NT/2), int(self.NR/2))))
        # self.gamma = nn.Parameter(torch.normal(1, 0.1, size=(1,self.NT,1)))

        self.theta1 = nn.Parameter(0.01 * torch.randn(1))
        self.gamma = nn.Parameter(torch.normal(1, 0.1, size=(1,self.NT,1)))

    def gaussian(self, zt, tau2_t):
        zt = zt
        #zt - symbols
        arg = torch.reshape(zt,[-1,1]).to(device=self.device) - self.lgst_constel.to(device=self.device)
        arg = torch.reshape(arg, [-1, self.NT, self.M]) 
        #-|| z - symbols||^2 / 2sigma^2
        arg = -torch.square(arg)/ 2. /  tau2_t
        arg = torch.reshape(arg, [-1, self.M]) 
        softMax = nn.Softmax(dim=1) 
        x_out = softMax(arg) 
        del arg
        # sum {xi exp()/Z}
        x_out = torch.matmul(x_out.double(), torch.reshape(self.lgst_constel, [self.M,1]).to(device=self.device).double()) 
        x_out = torch.reshape(x_out, [-1, self.NT])  
        return x_out
    
    

    def MMNet_denoiser(self, H, W, zt, xhat, rt, noise_sigma, batch_size):    
        HTH = torch.bmm(H.permute(0, 2, 1), H) 
        v2_t = torch.divide(torch.sum(torch.square(rt), dim=1, keepdim=True) - self.NR * torch.square(noise_sigma.unsqueeze(dim=1).to(device=self.device)) / 2, torch.unsqueeze(batch_trace(HTH), dim=1))
        v2_t = torch.maximum(v2_t , 1e-9 * torch.ones(v2_t.shape).to(device=self.device))
        v2_t = torch.unsqueeze(v2_t, dim=2)
        
        C_t = batch_identity_matrix(self.NT, H.shape[-1], batch_size).to(device=self.device) - torch.matmul(W.double(), H.double())
        tau2_t = 1./self.NT * torch.reshape(batch_trace(torch.bmm(C_t, C_t.permute(0, 2, 1)) ), [-1,1,1]) * v2_t + torch.square(torch.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * torch.reshape(batch_trace(torch.bmm(W, W.permute(0, 2, 1))), [-1,1,1])

        xhat = self.gaussian(zt, tau2_t/self.gamma.to(device=self.device))
        return xhat

    
    def MMNet_linear(self, H, y, xhat, batch_size):

        # W = torch.cat([torch.cat([self.Wr.to(device=self.device), -self.Wi.to(device=self.device)], dim=2),\
        #                torch.cat([self.Wi.to(device=self.device), self.Wr.to(device=self.device)], dim = 2)],\
        #                 dim =1)
        # W = W.repeat(batch_size, 1, 1)
        
        rt = y - batch_matvec_mul(H.double(), xhat.double())
        W = H.permute(0, 2, 1)
        zt = xhat + self.theta1 * batch_matvec_mul(W.double(), rt.double())

        return zt, rt, W


    
    def process_forward(self, H, y, x_out, noise_sigma, batch_size):

        zt, rt, W = self.MMNet_linear(H, y, x_out, batch_size)
        x_out = self.MMNet_denoiser(H, W, zt, x_out, rt, noise_sigma, batch_size)

        return x_out

    def forward(self, H, y, x_out, noise_sigma, batch_size):
        return self.process_forward(H, y, x_out, noise_sigma, batch_size)