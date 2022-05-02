"""
train.py training runner for U-langevin
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pickle as pkl

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from utils.util import *


def train(H_test, model, optimizer, generator, config, train_batch_size, test_batch_size, train_iter, device='cuda'):
    criterion = nn.MSELoss().to(device=device)
    model.train()
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)

    #######################
    ###-------SVD-------###
    #######################
    U, singulars_test, V = torch.svd(H_test)

    Uh_real_test = torch.transpose(U.to(device=device), 1, 2).to(device=device)
    Vh_real_test = torch.transpose(V.to(device=device), 1, 2).to(device=device)

    Sigma_test = torch.zeros((test_batch_size, 2 * config.NT, 2 * config.NT))
    for ii in range(test_batch_size):
        Sigma_test[ii,:, :] = torch.diag(singulars_test[ii,:])
        
    index = 0
    y_test, x_test, j_indices_test, noise_sigma = generator.give_batch_data_Hinput(H_test, config.NT, snr_db_min=config.SNR_dBs[config.NT][-1], 
                                                            snr_db_max=config.SNR_dBs[config.NT][-1], batch_size = test_batch_size)
    y_test = y_test.to(device=device).double()
    noise_sigma = noise_sigma.to(device=device).to(device=device).double()
    noise_sigma = torch.unsqueeze(noise_sigma, dim=-1)
    x_test = x_test.to(device=device)
    j_indices_test = j_indices_test.to(device=device)

    for i in range(train_iter):

#         print(i)
        ########################
        #--Samples generation--#
        ########################
        H, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT, snr_db_min=config.SNR_dBs[config.NT][0], 
                                                                    snr_db_max=config.SNR_dBs[config.NT][-1], batch_size=train_batch_size, 
                                                                    correlated_flag=config.corr_flag, rho=config.rho)        
        y = y.to(device=device).double()
        
        ######################
        ##-------SVD-------###
        #######################
        U, singulars, V = torch.svd(H)

        Uh_real = torch.transpose(U.to(device=device), 1, 2).to(device=device)
        Vh_real = torch.transpose(V.to(device=device), 1, 2).to(device=device)

        Sigma = torch.zeros((train_batch_size, 2 * config.NT, 2 * config.NT))
        for ii in range(train_batch_size):
            Sigma[ii,:, :] = torch.diag(singulars[ii,:])
   
        ########################
        ##---MMSE estimator---##
        ########################
#         y_MMSE = mmse(y.double().to(device=device), H.double().to(device=device), noise_sigma.double().to(device=device), device).double()

        ###############################
        ##----Langevin estimator----##
        ###############################
        noise_sigma = torch.unsqueeze(noise_sigma, dim=-1).to(device=device)
        sample_last, samples = model.forward(singulars.double(), Sigma.to(device=device).double(),
                                             Uh_real.double(), Vh_real.double(), y, noise_sigma.double(), config.NT, 
                                                config.M, config.temp)

        x = x.to(device=device).double()
        j_indices = j_indices.to(device=device).double()
       
        loss, SER, loss_level = loss_fn(x, samples, config.n_sigma_gaussian-1, j_indices, real_QAM_const, imag_QAM_const, criterion)                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del y, x, j_indices,  sample_last
        
        if (i%100==0):
            model.eval()

            with torch.no_grad():
                sample_last, samples = model.forward(singulars_test.double(), Sigma_test.to(device=device).double(),
                                                    Uh_real_test.double(), Vh_real_test.double(), y_test, noise_sigma, config.NT, 
                                                    config.M, config.temp)
               
                loss_last, SER_final, loss_level = loss_fn(x_test, samples,  config.n_sigma_gaussian-1, j_indices_test, real_QAM_const, imag_QAM_const, criterion)                
                results = [loss_last.detach().item(), 1 - SER_final, loss_level.detach().item()]
                print_string = [i]+results
                print(' '.join('%s' % np.round(x,6) for x in print_string))
                if (i%5000 == 0 and i>0):
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 1e-1
                    

            model.train()
            # torch.save(model.state_dict(), 'U_langevin_model')