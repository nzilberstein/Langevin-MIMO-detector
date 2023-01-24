################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
from asyncio import SubprocessTransport
from distutils.command.config import config
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl
import yaml
import os
from numpy import linalg as LA
import scipy.io as sio
import random

#\\Own libraries
from model.classic_detectors import *
from data.sample_generator import *
from utils.util import *
from runners.runner_langevin import *
from runners.runner_classic_methods import *
from pathlib import Path

################################################################################
#
####                              SETTINGS
#
################################################################################

######################
###  Load parameters of the system ###
######################
dirPath = str(Path(os.path.dirname(__file__)))
with open(dirPath + '/config.yml', 'r') as f:
    aux = yaml.load(f,  Loader=yaml.FullLoader)
config = dict2namespace(aux)


######################
###  General setup ###
######################
SEED = 123
torch.manual_seed(SEED)
useGPU = True # If true, and GPU is available, use it.
loadChannelInput = True
LANGEVIN_DETECTOR = True
CLASSICAL_DETECTORS = False

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = 'cuda'
else:
    device = 'cpu'

################################################################################
#
####                               DATA
#
################################################################################

def loadData():
    if loadChannelInput == True:
        ################
        ## Load Data ###
        ################

        #\\ IID channels
        # with open(dirPath + '/data/Hiid_5000bs_3264', 'rb') as fp:
        #     H = pkl.load(fp)
        # batch_size = H.shape[0]

        #\\ Kronecker channels
        with open(dirPath + '/data/H_5000bs_3264', 'rb') as fp:
            H = pkl.load(fp)
        batch_size = H.shape[0]

        #\\ 3gpp case
        # mat_contents = sio.loadmat(dirPath + '/data/H_bank.mat')
        # H = mat_contents['H_bank']
        # # H = torch.tensor(H[:, :, 0:config.NT])
        # H = torch.tensor(H[:, :, random.sample(range(100), config.NT)])#Pick up NT random users from 100.
        # batch_size = H.shape[0]
        # Hr = torch.real(H)
        # Hi = torch.imag(H)

        # h1 = torch.cat((Hr, -1. * Hi), dim=2)
        # h2 = torch.cat((Hi, Hr), dim=2)
        # H = torch.cat((h1, h2), dim=1)

    else:
        batch_size = 5000

    return batch_size, H


################################################################################
#
####                               MAIN
#
################################################################################


def main():
    #Create generator
    batch_size, H = loadData()
    if loadChannelInput == True:
        generator = sample_generator(batch_size, config.mod_n, config.NR)
    else:
        generator = sample_generator(batch_size, config.mod_n, config.NR)
        H, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT, 
                                                                snr_db_min=config.SNR_dBs[config.NT][0], snr_db_max=config.SNR_dBs[config.NT][0], 
                                                                batch_size=batch_size, correlated_flag=config.corr_flag, rho=config.rho)  

    if CLASSICAL_DETECTORS == True:
        serMMSE, serSDR = runClassicDetectors(config, generator, batch_size, device, H = H)
        with open(dirPath + '/results/MMSE_results', "wb") as output_file:
            pkl.dump(serMMSE, output_file)
        with open(dirPath + '/results/SDR_results', "wb") as output_file:
            pkl.dump(serSDR, output_file)
    if LANGEVIN_DETECTOR == True:
        serLangevin = runLangevin(config, generator, batch_size, device, H = H)
        with open(dirPath + '/results/langevin_results', "wb") as output_file:
            pkl.dump(serLangevin, output_file)

if __name__ == '__main__':
    main()
