################################################################################
#
####                               IMPORTING
#
################################################################################

#\\Standard libraries
from distutils.command.config import config
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl
import yaml
import os
from numpy import linalg as LA

#\\Own libraries
from model.classic_detectors import *
from data.sample_generator import *
from utils.util import *
from runner_langevin import *
from runner_classic_methods import *
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

#################
### Load Data ###
#################
with open(dirPath + '/data/H_5000bs_3264', 'rb') as fp:
    H = pkl.load(fp)
batch_size = H.shape[0]
   

def main():
    #Create generator
    generator = sample_generator(batch_size, config.mod_n, config.NR)

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
