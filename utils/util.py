from unicodedata import name
import torch
import numpy as np
from torchmetrics import SNR
import time 
import pickle as pkl
import random

"""
utils.py Utils functions

This class handle the sample generator module, such as the symbol detection
"""



def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
    #Convert to complex
    x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
    #Expand to the constellation size
    x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
    x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

    #Difference w.r.t. to each symbol
    x_real = torch.pow(x_real - real_QAM_const, 2)
    x_imag = torch.pow(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    x_indices = torch.argmin(x_dist, dim=-1)

    accuracy = (x_indices == j_indices).sum().to(dtype=torch.float32)
    return accuracy.item()/x_indices.numel()


def batch_matvec_mul(A,b):
    '''Multiplies a matrix A of size batch_sizexNxK
       with a vector b of size batch_sizexK
       to produce the output of size batch_sizexN
    '''    
    C = torch.matmul(A, torch.unsqueeze(b, dim=2))
    return torch.squeeze(C, -1) 

def batch_identity_matrix(row, cols, batch_size):
    eye = torch.eye(row, cols)
    eye = eye.reshape((1, row, cols))
    
    return eye.repeat(batch_size, 1, 1)

def dict2namespace(config):
    namespace = type('new', (object,), config)
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    
    namespace = addAttr(namespace)

    return namespace

def addAttr(config):
    M = int(np.sqrt(config.mod_n))
    sigConst = np.linspace(-M+1, M-1, M) 
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.) 
    setattr(config, 'M', M)
    setattr(config, 'sigConst', sigConst)
    
    M2 = int(np.sqrt(config.mod_n2))
    sigConst2 = np.linspace(-M2+1, M2-1, M2) 
    sigConst2 /= np.sqrt((sigConst2 ** 2).mean())
    sigConst2 /= np.sqrt(2.) 
    setattr(config, 'M2', M2)
    setattr(config, 'sigConst2', sigConst2)
    
    SNR_dBs = {config.NT :np.arange( config.SNR_db_min, config.SNR_db_max)}
    setattr(config, 'SNR_dBs', SNR_dBs)
    
    SNR_dBs_two_mod = {config.NT :np.arange( config.SNR_db_min, config.SNR_db_max_two_mod)}
    setattr(config, 'SNR_dBs_two_mod', SNR_dBs_two_mod)
    return config


def loss_fn(x, list_batch_x_predicted, num_layers, j_indices, real_QAM_const, imag_QAM_const, criterion, ser_only=False):
    x_out = torch.cat(list_batch_x_predicted, dim=0)
    loss_last = criterion(list_batch_x_predicted[-1].double(), x.double())
    x = x.repeat(num_layers, 1)
    loss = criterion(x_out.double(), x.double())
    SER_final = sym_detection(list_batch_x_predicted[-1].double(), j_indices, real_QAM_const, imag_QAM_const)
    return loss, SER_final, loss_last


def model_eval(model, generator, config, test_batch_size, n_traj, device):

    SER_lang32u = []
    H_test, y, x, j_indices, noise_sigma = generator.give_batch_data(config.NT, snr_db_min=config.SNR_dBs[config.NT][0], snr_db_max=config.SNR_dBs[config.NT][0], 
                                                                batch_size=test_batch_size, correlated_flag=config.corr_flag, rho=config.rho)        
    y = y.to(device=device).double()

    U, singulars, V = torch.svd(H_test)

    Uh_real = torch.transpose(U.to(device=device), 1, 2).to(device=device)
    Vh_real = torch.transpose(V.to(device=device), 1, 2).to(device=device)

    Sigma = torch.zeros((test_batch_size, 2 * config.NT, 2 * config.NT))
    for ii in range(test_batch_size):
        Sigma[ii,:, :] = torch.diag(singulars[ii,:])
            
    noise_sigma = torch.unsqueeze(noise_sigma, dim=-1).to(device=device)

    for snr in range(config.SNR_dBs[config.NT]):
        print(config.SNR_dBs[config.NT][snr])
        # Create variables to save each trajectory and for each snr
        dist = torch.zeros((test_batch_size,config.n_traj))
        list_traj = torch.zeros((test_batch_size, 2*config.NT, config.n_traj))
        x_hat = torch.zeros((test_batch_size, 2*config.NT))


        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H_test, config.NT, snr_db_min=config.SNR_dBs[config.NT][snr], 
                                                                        snr_db_max=config.SNR_dBs[config.NT][snr], batch_size = test_batch_size)

        with torch.no_grad():
            for jj in range(0, n_traj):
                start = time.time()

                sample_last, samples = model.forward(singulars.double(), Sigma.to(device=device).double(),
                                                            Uh_real.double(), Vh_real.double(), y, noise_sigma, test_batch_size,
                                                            config.step_size, config.temp)
                list_traj[:,:,jj] = torch.clone(sample_last.to(device=device))
                dist[:, jj] = torch.norm(y.to(device=device).float() - 
                                        batch_matvec_mul(H_test.to(device=device).float(), sample_last.to(device=device.float()), 2, dim = 1))
                end = time.time()
                print(end - start)
                torch.cuda.empty_cache()

            idx = torch.argsort(dist, dim=1, descending = False)

            for nnn in range(0, test_batch_size):
                x_hat[nnn, :] = torch.clone(list_traj[nnn,:,idx[nnn,0]])

        # Detection   
        SER_lang32u.append(1 - sym_detection(x_hat.to(device=device), j_indices, generator.real_QAM_const, generator.imag_QAM_const))
        print(SER_lang32u)

    return SER_lang32u


def loadData(ofdmSymb, subcarrier,subcarrierIndex, int_min, int_max, dirPath, dataIndex, config):

    with open(dirPath + '/data/dataPilot', 'rb') as fp:
        pilotIndex, pilotValue = pkl.load(fp)
        # pilotIndex = [3, 5, 10, 15, 22]


    with open(dirPath + '/data/Hreal_MMSE', 'rb') as fp:
        HH = pkl.load(fp)
        HPilot = HH[int_min:int_max,:,:,[pilotIndex[0] - 1, pilotIndex[1] - 1, 31, 45]]
        # HPilot = H[int_min:int_max,:,:,[x - 1 for x in pilotIndex]]
        H = np.zeros((HH.shape[0], HH.shape[1], HH.shape[2], len(dataIndex)), dtype = 'complex')
        index = 0
        for ii in range(HH.shape[-1]):
            if (ii == pilotIndex[0] - 1) or (ii == pilotIndex[1] - 1) or ii == 31 or ii == 45:
                # print(ii)
                continue
                
            else:
                H[int_min:int_max,:,:,index] = HH[int_min:int_max,:,:,ii]    
                index = index + 1

        H = H[int_min:int_max,:,:,subcarrierIndex]

    with open(dirPath + '/data/Yreal_MMSE', 'rb') as fp:
        y = pkl.load(fp)
        #-,odfmsym,-,subcarrier
        yPilot = y[int_min:int_max,ofdmSymb,:,pilotIndex]
        # yPilot = np.transpose(yPilot, (1, 2, 0))
        y = y[int_min:int_max,ofdmSymb,:,subcarrier]
        
    with open(dirPath + '/data/Xreal_MMSE', 'rb') as fp:
        x = pkl.load(fp)
        # -,-,-,odfmsym,subcarrier
        xPilot = x[int_min:int_max,:,0,ofdmSymb,pilotIndex]
        # x = x[int_min:int_max,:,0,0,7]
        x = x[int_min:int_max,:,0,ofdmSymb,subcarrier]

    with open(dirPath + '/data/variancereal', 'rb') as fp:
        sigma = pkl.load(fp)
        # sigma = torch.tensor(sigma)
        # print(sigma)
        sigma = torch.tensor(1e-01)

    with open(dirPath + '/data/snrreal', 'rb') as fp:
        snr = pkl.load(fp)
        snr = snr[:,0,:,:]


    with open(dirPath + '/data/dataPilot', 'rb') as fp:
        pilot_index, pilot_val = pkl.load(fp)


    batch_size = H.shape[0]

    H = torch.tensor(H)
    Hr = torch.real(H)
    Hi = torch.imag(H)

    h1 = torch.cat((Hr, 1. * Hi), dim=2)
    h2 = torch.cat((-1. * Hi, Hr), dim=2)
    H = torch.cat((h1, h2), dim=1)
    H = H.permute(0,2,1).double()
    H_MMSE = H
    H_comp = H[:, 0:config.NR, 0:config.NT] + 1j*H[:, 0:config.NR, config.NT:]

    Hp = torch.tensor(HPilot)
    Hr = torch.real(Hp)
    Hi = torch.imag(Hp)

    h1 = torch.cat((Hr, 1. * Hi), dim=2)
    h2 = torch.cat((-1. * Hi, Hr), dim=2)
    HPilotri = torch.cat((h1, h2), dim=1)
    HPilotri = HPilotri.permute(0,2,1,3).double()

    y = torch.tensor(y)
    yr = torch.real(y)
    #The sign is because in the original data it is a row vector, and here we use as a column vector. So we need to transpose
    #and therefore conjugate.
    yi = torch.imag(y)

    yri = torch.cat((yr, yi), dim=1)

    y = torch.tensor(yPilot)
    yr = torch.real(y)
    yi = torch.imag(y)

    yPilotri = torch.cat((yr, yi), dim=2)


    x = torch.tensor(x)
    xr = torch.real(x)
    xi = torch.imag(x)

    xri = torch.cat((xr, xi), dim=1)

    x = torch.tensor(xPilot)
    xr = torch.real(x)
    xi = torch.imag(x)

    xPilotri = torch.cat((xr, xi), dim=1)

    # phaseCorr = torch.tensor(phaseCorr)
    # pcr = torch.real(phaseCorr)
    # pci = torch.imag(phaseCorr)

    # phaseCorrri = torch.cat((pcr, pci), dim=1)

    return batch_size, H_MMSE, H_comp, HPilotri, yPilotri, xri, xPilotri, pilotValue, pilotIndex, sigma, yri

def sym_detection_absolute(x_hat, j_indices, real_QAM_const, imag_QAM_const):
    #Convierte a complejo
    x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
    #Lo expande a los 4 posibles simbolos para comparar
    x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
    x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

    #Calcula la resta
    x_real = torch.pow(x_real - real_QAM_const, 2)
    x_imag = torch.pow(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    x_indices = torch.argmin(x_dist, dim=-1)

    accuracy = (x_indices == j_indices).sum().to(dtype=torch.float32)
    return accuracy.item()


def loadRealChannels(dirPath, int_min, int_max):

    with open(dirPath + '/data/dataPilot', 'rb') as fp:
        pilotIndex, pilotValue = pkl.load(fp)

    with open(dirPath + '/data/dataIndex', 'rb') as fp:
        dataIndex = pkl.load(fp) 

    with open(dirPath + '/data/Hreal_MMSE', 'rb') as fp:
        HH = pkl.load(fp)
        HPilot = HH[int_min:int_max,:,:,[pilotIndex[0] - 1, pilotIndex[1] - 1, 31, 45]]
        # HPilot = H[int_min:int_max,:,:,[x - 1 for x in pilotIndex]]
        H_aux = np.zeros((HH.shape[0] * len(dataIndex), HH.shape[1], HH.shape[2]), dtype = 'complex')
        index = 0
        for ii in range(len(dataIndex)):
            if (ii == pilotIndex[0] - 1) or (ii == pilotIndex[1] - 1) or ii == 31 or ii == 45:
                continue
            else:
                H_aux[int_min + ii * int_max:int_max + ii * int_max,:,:] = HH[int_min:int_max,:,:,ii]    
                index = index + 1

        # H = torch.tensor( H_aux[random.sample(range(HH.shape[0] * len(dataIndex)), 1000), :, :] )
        H = torch.tensor( H_aux[0:800, :, :] )
    batch_size = H.shape[0]
    Hr = torch.real(H)
    Hi = torch.imag(H)

    h1 = torch.cat((Hr, -1. * Hi), dim=2)
    h2 = torch.cat((Hi, Hr), dim=2)
    H = torch.cat((h1, h2), dim=1)

    H = H.permute(0,2,1).double()

    return H, batch_size