"""
sample_generator_twomodulations.py Sample generator class for generations two modulations

This class handle the sample generator module

Remark.
There are two methods to generate the data: 
    1) give_batch_data is a method that generate both channel and observation (as well as the true symbols)    
    1) give_batch_data_Hinput is a method that generate observations based on a batch of input (as well as the true symbols) 
"""



import torch
import numpy as np
from numpy import linalg as LA
from scipy.stats import ortho_group as og
import scipy.linalg as LA

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/utils')
from util import *
from sample_generator import *


def adjust_var(data, NR):
    factor = 1.0/np.sqrt(2.0*NR*data.var())
    return data*factor

class sample_generator_twomods(object):
    def __init__(self, batch_size, mod_n1, mod_n2, NR):
        self.generator1 = sample_generator(batch_size, mod_n1, NR)
        self.generator2 = sample_generator(batch_size, mod_n2, NR)
        self.NR = NR
        self.mod_n1 = mod_n1
        self.mod_n2 = mod_n2
        self.batch_size = batch_size
        self.Hdataset_powerdB = np.inf

    def batch_exp_correlation(self, rho_low, rho_high, batch_size, NT):
        ranger = np.reshape(np.arange(1, self.NR+1), (-1,1))
        ranget = np.reshape(np.arange(1, NT+1), (-1,1))
        rho_list = np.random.uniform(rho_low, rho_high, size=(batch_size))

        Rr_list = [rho_list[i]**(np.abs(ranger - ranger.T)) for i in range(batch_size)]
        Rt_list = [rho_list[i]**(np.abs(ranget - ranget.T)) for i in range(batch_size)]

        R1 = np.asarray([LA.sqrtm(Rr_list[i]) for i in range(batch_size)])
        R2 = np.asarray([LA.sqrtm(Rt_list[i]) for i in range(batch_size)])

        R1 = torch.from_numpy(R1).to(dtype=torch.float32)
        R2 = torch.from_numpy(R2).to(dtype=torch.float32)
        return R1, R2

    def exp_correlation(self, rho, batch_size, NT):
        ranger = np.reshape(np.arange(1, self.NR+1), (-1,1))
        ranget = np.reshape(np.arange(1, NT+1), (-1,1))
        Rr = rho ** (np.abs(ranger - ranger.T))
        Rt = rho ** (np.abs(ranget - ranget.T))
        R1 = LA.sqrtm(Rr)
        R2 = LA.sqrtm(Rt)
        R1 = torch.from_numpy(R1).to(dtype=torch.float32)
        R1 = R1.expand(size=(batch_size, -1, -1))
        R2 = torch.from_numpy(R2).to(dtype=torch.float32)
        R2 = R2.expand(size=(batch_size, -1, -1))
        return R1, R2

    def channel(self, x, snr_db_min, snr_db_max, NT, batch_size, correlated_flag, rho, batch_corr, rho_low, rho_high, QR, Cu):

        if (QR):
            Q,R,H_true = createQR(Cu, self.batch_size)
            H = torch.tensor(R)
            H_powerdB = 10. * torch.log(torch.mean(torch.sum(H.pow(2), dim=1), dim=0)) / np.log(10.)
#             self.Hdataset_powerdB = torch.mean(H_powerdB)
            self.Hdataset_powerdB = 0.

        else:
            Hr = torch.empty((batch_size, self.NR, NT)).normal_(mean=0,std=1./np.sqrt(2.*self.NR))
            Hi = torch.empty((batch_size, self.NR, NT)).normal_(mean=0,std=1./np.sqrt(2.*self.NR))

            if (correlated_flag):
                if (batch_corr):
                    R1, R2 = self.batch_exp_correlation(rho_low, rho_high, batch_size, NT)
                    Hr = torch.einsum(('bij,bjl,blk->bik'), (R1, Hr, R2))
                    Hi = torch.einsum(('bij,bjl,blk->bik'), (R1, Hi, R2))		
                else:
                    R1, R2 = self.exp_correlation(rho, batch_size, NT)
                    Hr = torch.einsum(('bij,bjl,blk->bik'), (R1, Hr, R2))
                    Hi = torch.einsum(('bij,bjl,blk->bik'), (R1, Hi, R2))
#             Q, R = torch.qr(Hr + 1j*Hi)
#             h1 = torch.cat((torch.real(R), -1. * torch.imag(R)), dim=2)
#             h2 = torch.cat((torch.imag(R), torch.real(R)), dim=2)
#             H = torch.cat((h1, h2), dim=1)
            h1 = torch.cat((Hr, -1. * Hi), dim=2)
            h2 = torch.cat((Hi, Hr), dim=2)
            H = torch.cat((h1, h2), dim=1)
            self.Hdataset_powerdB = 0.

        # Channel Noise
        snr_db = torch.empty((batch_size, 1)).uniform_(snr_db_min, snr_db_max)

        wr = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        wi = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        w = torch.cat((wr, wi), dim=1)

        # SNR
        H_powerdB = 10. * torch.log(torch.mean(torch.sum(H.pow(2), dim=1), dim=0)) / np.log(10.)
        average_H_powerdB = torch.mean(H_powerdB)
        average_x_powerdB = 10. * torch.log(torch.mean(torch.sum(x.pow(2), dim=1))) / np.log(10.)

        w *= torch.pow(10., (10.*np.log10(NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)
        complexnoise_sigma = torch.pow(10., (10.*np.log10(NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)

        # Channel Output
        y = batch_matvec_mul(H.double(), x.double()) + w.double()
        sig_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(self.batch_matvec_mul(H.double(),x.double()),2), dim=1))) / np.log(10.)
        noise_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(w,2), axis=1))) / np.log(10.)
        actual_snrdB = sig_powdB - noise_powdB

        return y, H, complexnoise_sigma, actual_snrdB

    def channel_Hinput(self, H, x, snr_db_min, snr_db_max, NT, batch_size):
        
        self.Hdataset_powerdB = 0.
        
        # Channel Noise
        snr_db = torch.empty((batch_size, 1)).uniform_(snr_db_min, snr_db_max)

        wr = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        wi = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        w = torch.cat((wr, wi), dim=1)

        # SNR
        H_powerdB = 10. * torch.log(torch.mean(torch.sum(H.pow(2), dim=1), dim=0)) / np.log(10.)
        average_H_powerdB = torch.mean(H_powerdB)
        average_x_powerdB = 10. * torch.log(torch.mean(torch.sum(x.pow(2), dim=1))) / np.log(10.)

        w *= torch.pow(10., (10.*np.log10(NT) + average_H_powerdB - snr_db - 10.*np.log10(self.NR))/20.)
        complexnoise_sigma = torch.pow(10., (10.*np.log10(NT) + average_H_powerdB  - snr_db - 10.*np.log10(self.NR))/20.)

        w = w.to(device='cuda')
        x = x.to(device='cuda')
        H = H.to(device='cuda')
        # Channel Output
        y = batch_matvec_mul(H.double(), x.double()) + w.double()
        sig_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(batch_matvec_mul(H.double(),x.double()),2), dim=1))) / np.log(10.)
        noise_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(w,2), axis=1))) / np.log(10.)
        actual_snrdB = sig_powdB - noise_powdB

        return y, complexnoise_sigma, actual_snrdB


    def ortho_channel(self, x, snr_db_min, snr_db_max, NT, batch_size):

        H = torch.empty(batch_size, 2*self.NR, 2*NT)

        for i in range(batch_size):

            ortho_matrix = og.rvs(self.NR)

            Hr = ortho_matrix[:,:NT]
            Hi = ortho_matrix[:,NT:2*NT]

            Hr = adjust_var(Hr, self.NR)
            Hi = adjust_var(Hi, self.NR)

            Hr = torch.from_numpy(Hr).to(dtype=torch.float32)
            Hi = torch.from_numpy(Hi).to(dtype=torch.float32)

            h1 = torch.cat((Hr, -1.*Hi), dim=-1)
            h2 = torch.cat((Hi, Hr), dim=-1)

            h = torch.cat((h1,h2), dim=0)
            H[i] = h

        self.Hdataset_powerdB = 0.

        snr_db = torch.empty((batch_size, 1)).uniform_(snr_db_min, snr_db_max)

        wr = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        wi = torch.empty((batch_size, self.NR)).normal_(mean=0.0, std=1./np.sqrt(2.))
        w = torch.cat((wr, wi), dim=1)

        # SNR
        H_powerdB = 10. * torch.log(torch.mean(torch.sum(H.pow(2), dim=1), dim=0)) / np.log(10.)
        average_H_powerdB = torch.mean(H_powerdB)
        average_x_powerdB = 10. * torch.log(torch.mean(torch.sum(x.pow(2), dim=1))) / np.log(10.)

        w *= torch.pow(10., (10.*np.log10(NT) + average_H_powerdB - snr_db - 10.*np.log10(self.NR))/20.)
        complexnoise_sigma = torch.pow(10., (10.*np.log10(NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)

        # Channel Output
        y = self.batch_matvec_mul(H, x) + w
        sig_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(self.batch_matvec_mul(H,x),2), dim=1))) / np.log(10.)
        noise_powdB = 10. * torch.log(torch.mean(torch.sum(torch.pow(w,2), axis=1))) / np.log(10.)
        actual_snrdB = sig_powdB - noise_powdB

        return y, H, complexnoise_sigma, actual_snrdB


    def give_batch_data(self, NT, snr_db_min=2, snr_db_max=7, batch_size=None, correlated_flag=False, rho=None, batch_corr=False, rho_low=None, rho_high=None, QR=None, Cu=None):
        if (batch_size==None):
            batch_size = self.batch_size
        indices1 = self.generator1.random_indices(int(NT/2), batch_size)
        joint_indices1 = self.generator1.joint_indices(indices1)
        x1 = self.generator1.modulate(indices1)
        
        indices2 = self.generator2.random_indices(int(NT/2), batch_size)
        joint_indices2 = self.generator2.joint_indices(indices2)
        x2 = self.generator2.modulate(indices2)
        
        x = torch.cat((x1[:,0:int(NT/2)], x2[:,0:int(NT/2)], x1[:,int(NT/2):], x2[:,int(NT/2):]), dim=1)
        joint_indices = torch.cat((joint_indices1, joint_indices2), dim=1)

        y, H, complexnoise_sigma, _ = self.channel(x, snr_db_min, snr_db_max, NT, batch_size, correlated_flag, rho, batch_corr, rho_low, rho_high, QR, Cu)
        return H, y, x, joint_indices, complexnoise_sigma.squeeze()
    
    def give_batch_data_Hinput(self, H, NT, snr_db_min=2, snr_db_max=7, batch_size=None):
        if (batch_size==None):
            batch_size = self.batch_size
        indices1 = self.generator1.random_indices(int(NT/2), batch_size)
        joint_indices1 = self.generator1.joint_indices(indices1)
        x1 = self.generator1.modulate(indices1)
        
        indices2 = self.generator2.random_indices(int(NT/2), batch_size)
        joint_indices2 = self.generator2.joint_indices(indices2)
        x2 = self.generator2.modulate(indices2)
        
        x = torch.cat((x1[:,0:int(NT/2)], x2[:,0:int(NT/2)], x1[:,int(NT/2):], x2[:,int(NT/2):]), dim=1)
        joint_indices = torch.cat((joint_indices1, joint_indices2), dim=1)

        y, complexnoise_sigma, _ = self.channel_Hinput(H, x, snr_db_min, snr_db_max, NT, batch_size)
        return y, x, joint_indices, complexnoise_sigma.squeeze()
    
    def analyze_batch_data(self, NT, snr_db_min, snr_db_max, batch_size=None):
        if (batch_size==None):
            batch_size = self.batch_size
        indices = self.random_indices(NT, batch_size)
        joint_indices = self.joint_indices(indices)
        x = self.modulate(indices)
        y, H, complexnoise_sigma, _ = self.ortho_channel(x, snr_db_min, snr_db_max, NT, batch_size)
        return H, y, joint_indices, complexnoise_sigma.squeeze()