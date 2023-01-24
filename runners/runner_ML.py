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
from model.ML import *
from data.sample_generator import *
from utils.util import *
from multiprocessing.dummy import Pool as ThreadPool 


################################################################################
#
####                               MAIN RUN
#
################################################################################

def runML(config, generator, batch_size, device, H = None):

    #Define list to save data
    SER_ML = []

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

        ###############################
        ##  ML  ##
        ###############################

        pool = ThreadPool(40) 
        x_ml = pool.map(ml_proc_star, zip(H.cpu().detach().numpy(), y.unsqueeze(dim=-1).cpu().detach().numpy()))
        x_ml = np.array(x_ml).squeeze(axis=1)
        SER_ML.append(1 - sym_detection(torch.from_numpy(x_ml), 
                            j_indices, generator.real_QAM_const, generator.imag_QAM_const))

        print(SER_ML)

    return SER_ML
