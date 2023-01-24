################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
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
from model.langevin import *


################################################################################
#
####                               MAIN RUN
#
################################################################################
def runLangevin(config, generator, batch_size, device, H = None, y = None, noise_sigma = None, j_indices = None):
    """
    Runner for langevin
    Input:
        config: class that contain all the settings
        generator: generator class, used to generate data
        batch_size: As we don't need to have a channel as input, we can pre-specified the batch_size
        step: epsilon step size of the algorithm
    
    """
    #########################################################
    ## Variables definition ## 
    #########################################################

    #Create noise vector for Langevin
    sigma_gaussian = np.exp(np.linspace(np.log(config.sigma_0), np.log(config.sigma_L),config.n_sigma_gaussian))

    #Define list to save data
    SER_lang32u = []

    #Create model
    langevin = Langevin(sigma_gaussian, generator, config.n_sample_init, config.step_size, device)

    #############
    ###  SVD  ###
    #############
    U, singulars, V = torch.svd(H)

    Uh_real = torch.transpose(U.to(device=device), 1, 2).to(device=device)
    Vh_real = torch.transpose(V.to(device=device), 1, 2).to(device=device)

    Sigma = torch.zeros((batch_size, 2 * config.NT, 2 * config.NT))
    for ii in range(batch_size):
        Sigma[ii,:, :] = torch.diag(singulars[ii,:])


    #########################################################
    ## Main loop ## 
    #########################################################

    for snr in range(0, len(config.SNR_dBs[config.NT])):
        print(config.SNR_dBs[config.NT][snr])
        # Create variables to save each trajectory and for each snr
        dist = torch.zeros((batch_size,config.n_traj))
        list_traj = torch.zeros((batch_size, 2*config.NT, config.n_traj))
        x_hat = torch.zeros((batch_size, 2*config.NT))

        ########################
        #  Samples generation  #
        ########################
        #Generate data
    #     H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=SNR_dBs[NT][snr], snr_db_max=SNR_dBs[NT][snr], batch_size=batch_size, correlated_flag=corr_flag, rho=rho)        

        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr],
                                                                    snr_db_max=config.SNR_dBs[config.NT][snr], 
                                                                    batch_size = batch_size)

        # print(noise_sigma[0], singulars * config.sigma_L)
        y = y.to(device=device)
        
        
        ###############################
        ##  Langevin detector  ##
        ###############################
        for jj in range(0, config.n_traj):
            #start = time.time()
            #run langevin
            sample_last, samples = langevin.forward(singulars.float(), Sigma.to(device=device).float(), 
                                    Uh_real.float(), Vh_real.float(), y, noise_sigma[0], 
                                    config.NT, config.M, config.temp)
            
            #Generate n_traj realizations of Langevin and then choose the best one w.r.t to ||y-Hx||^2
            list_traj[:,:,jj] = torch.clone(sample_last)
            dist[:, jj] = torch.norm(y.to(device=device) - batch_matvec_mul(H.to(device=device).float(),
                                                             sample_last.to(device=device)), 2, dim = 1)
            #end = time.time()
            print('Trajectory:', jj)

        #Pick the best trajectory for each user
        idx = torch.argsort(dist, dim=1, descending = False)

        for nn in range(0, batch_size):
            x_hat[nn, :] = torch.clone(list_traj[nn,:,idx[nn,0]])

        #Evaluate performance of Langevin detector
        SER_lang32u.append(1 - sym_detection(x_hat.to(device='cpu'), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
        print(SER_lang32u)

    return SER_lang32u