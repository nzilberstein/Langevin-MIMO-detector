import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from util import *
import yaml
import os
from pathlib import Path
plt.switch_backend('agg')
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

# with open(dirPath + '/results/iid_channel/oampnet_results', "rb") as input_file:
#     serOAMPNet = pkl.load(input_file)
with open(dirPath + '/results/MMSE_results', "rb") as input_file:
    serMMSE = pkl.load(input_file)
# with open(dirPath + '/results/iid_channel/SDR_results', "rb") as input_file:
#     serVBLAST = pkl.load(input_file)
with open(dirPath + '/results/ML_results', "rb") as input_file:
    serML = pkl.load(input_file)
# with open(dirPath + '/results/iid_channel/mmnet_results', "rb") as input_file:
#     serMMNet = pkl.load(input_file)


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


# plt.semilogy(config.SNR_dBs[config.NT][0:4],  serLangevin_20traj_T2[0:4], label= str(chr(964)) + ' = 2', marker = '*')
# plt.semilogy(config.SNR_dBs[config.NT][0:4],  serLangevin_20traj_T1[0:4], label= str(chr(964)) + ' = 1', marker = '*')
plt.semilogy(config.SNR_dBs[config.NT],  serLangevin, label = 'Langevin', marker = '*')
# plt.semilogy(config.SNR_dBs[config.NT][0:4],  serLangevin_20traj_T005[0:4], label= str(chr(964)) + ' = 0.05', marker = '*')
plt.semilogy(config.SNR_dBs[config.NT], serMMSE, label= 'MMSE', marker = 'x')
serML[9] = 0
plt.semilogy(config.SNR_dBs[config.NT], serML, label= 'ML', marker = '<')

plt.grid(True, which="both")
plt.legend(loc = 1, fontsize=15)
plt.xlabel('SNR', fontsize=14)
plt.ylabel('SER', fontsize=14)
plt.tick_params(axis='both' , labelsize=14)
plt.tight_layout()
plt.savefig(dirPath + '/results/langevin_real_channel.pdf')