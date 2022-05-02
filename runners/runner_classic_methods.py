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

def runClassicDetectors(config, generator, batch_size, device, H = None, y = None, noise_sigma = None, j_indices = None):

    #Define list to save data
    SER_BLAST = []
    SER_MMSE32u = []

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

        SER_MMSE32u.append(1 - sym_detection(x_MMSE.to(device='cpu'), j_indices, generator.real_QAM_const, 
                            generator.imag_QAM_const))
        
        
        ###############################
        ##  V-BLAST  ##
        ###############################


        x_blast = blast_eval(y.unsqueeze(dim=-1).cpu().detach().numpy(), H.cpu().detach().numpy(), 
                        config.sigConst, config.NT, config.NR).squeeze()
                        
        SER_BLAST.append(1 - sym_detection(torch.from_numpy(x_blast), 
                            j_indices, generator.real_QAM_const, generator.imag_QAM_const))

        print(SER_MMSE32u, SER_BLAST)

    return SER_MMSE32u, SER_BLAST
