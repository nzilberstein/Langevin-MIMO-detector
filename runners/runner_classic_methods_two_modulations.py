################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
from os import sched_get_priority_min
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import math
import sys
import pickle as pkl
from numpy import linalg as LA
import os

sys.path.append(os.path.dirname(os.getcwd()))

#\\Own libraries
from model.classic_detectors import *
from data.sample_generator import *
from utils.util import *



################################################################################
#
####                               MAIN RUN
#
################################################################################

def runClassicDetectors(config, generator, batch_size, device, H = None):

    #Define list to save data
    SER_lang32u_mod16 = []
    SER_lang32u_mod64 = []
    
    #########################################################
    ## Main loop ## 
    #########################################################

    for snr in range(0, len(config.SNR_dBs[config.NT])):
        print(config.SNR_dBs[config.NT][snr])


        ########################
        #  Samples generation  #
        ########################
    #     H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=SNR_dBs[NT][snr], snr_db_max=SNR_dBs[NT][snr], batch_size=batch_size, correlated_flag=corr_flag, rho=rho)        
    #     H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=SNR_dBs[NT][snr], snr_db_max=SNR_dBs[NT][snr], batch_size=batch_size)        
        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
                                                         snr_db_max=config.SNR_dBs[config.NT][snr], batch_size = batch_size)
        y = y.to(device=device)
        
        ########################
        ##  MMSE detector  ##
        ########################

        x_MMSE = mmse(y.float().to(device=device), H.float().to(device=device), 
                        noise_sigma.float().to(device=device), device).double()

        
        #Evaluate performance of MMSE detector
        x_hat_mod16 = torch.cat((x_MMSE[:,0:int(config.NT/2)], x_hat[:,config.NT: config.NT + int(config.NT/2)]), dim = 1)
        x_hat_mod64 = torch.cat((x_MMSE[:,int(config.NT/2):config.NT], x_hat[:,config.NT + int(config.NT/2):]), dim = 1)

        SER_lang32u_mod16.append((1 - sym_detection(x_hat_mod16.to(device='cpu'), j_indices[:, 0:int(config.NT/2)], generator.generator1.real_QAM_const, generator.generator1.imag_QAM_const)))
        SER_lang32u_mod64.append((1 - sym_detection(x_hat_mod64.to(device='cpu'), j_indices[:, int(config.NT/2):], generator.generator2.real_QAM_const, generator.generator2.imag_QAM_const)))

        print(SER_lang32u_mod16, SER_lang32u_mod64)
        print(SER_MMSE32u, SER_BLAST)

    return SER_MMSE32u, SER_BLAST
