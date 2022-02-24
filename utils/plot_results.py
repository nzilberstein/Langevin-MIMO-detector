import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from util import *
import yaml
import os
from pathlib import Path

######################################
###  Load parameters of the system ###
######################################

dirPath = str(Path(os.path.dirname(__file__)).parent.absolute())

with open(dirPath + '/config.yml', 'r') as f:
    aux = yaml.load(f,  Loader=yaml.FullLoader)
config = dict2namespace(aux)

######################
###  Load resutls ####
######################

"You will need to generate the results and then load using the following"
"command. Repeat it for the different cases."
with open(dirPath + '/results/langevin_results', "rb") as input_file:
    serLangevin = pkl.load(input_file)


################################################################################
#
####                               PLOTS
#
################################################################################


###############################
### Comparison trajectories ###
###############################

# plt.semilogy(config.SNR_dBs[config.NT], lang_true_1traj, label= '1 trajectory', marker = '*')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_5traj, label= '5 trajectories', marker = 'x')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_10traj, label= '10 trajectories' , marker = 'o')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_20traj , label= '20 trajectories', marker = '.')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_40traj , label= '40 trajectories', marker = 'v')

# plt.grid(True, which="both")
# plt.legend(loc = 1, fontsize=15)
# plt.ylim([1e-5, 3.2e-1])
# plt.xlabel('SNR', fontsize=14)
# plt.ylabel('SER', fontsize=14)
# plt.tick_params(axis='both' , labelsize=14)
# # plt.title('Nu = 32, Nr = 64, QAM-16')
# plt.savefig('langevin_comparison_traj.pdf')



###############################
### Comparison noise levels ###
###############################


# plt.semilogy(config.SNR_dBs[config.NT], lang_true_1traj, label= '1 trajectory', marker = '*')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_5traj, label= '5 trajectories', marker = 'x')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_10traj, label= '10 trajectories' , marker = 'o')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_20traj , label= '20 trajectories', marker = '.')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_40traj , label= '40 trajectories', marker = 'v')

# plt.grid(True, which="both")
# plt.legend(loc = 1, fontsize=15)
# plt.ylim([1e-5, 3.2e-1])
# plt.xlabel('SNR', fontsize=14)
# plt.ylabel('SER', fontsize=14)
# plt.tick_params(axis='both' , labelsize=14)
# # plt.title('Nu = 32, Nr = 64, QAM-16')
# plt.savefig('langevin_comparison_traj.pdf')

##########################
### Comparison methods ###
##########################


# plt.semilogy(config.SNR_dBs[config.NT], lang_true_1traj, label= '1 trajectory', marker = '*')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_5traj, label= '5 trajectories', marker = 'x')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_10traj, label= '10 trajectories' , marker = 'o')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_20traj , label= '20 trajectories', marker = '.')
# plt.semilogy(config.SNR_dBs[config.NT], lang_true_40traj , label= '40 trajectories', marker = 'v')

# plt.grid(True, which="both")
# plt.legend(loc = 1, fontsize=15)
# plt.ylim([1e-5, 3.2e-1])
# plt.xlabel('SNR', fontsize=14)
# plt.ylabel('SER', fontsize=14)
# plt.tick_params(axis='both' , labelsize=14)
# # plt.title('Nu = 32, Nr = 64, QAM-16')
# plt.savefig('langevin_comparison_traj.pdf')