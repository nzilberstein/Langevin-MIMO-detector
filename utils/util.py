from unicodedata import name
import torch
import numpy as np
from numpy import linalg as LA
from torchmetrics import SNR

"""
utils.py Utils functions

This class handle the sample generator module, such as the symbol detection
"""



def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
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
    
    SNR_dBs = {config.NT :np.arange( config.SNR_db_min, config.SNR_db_max)}
    setattr(config, 'SNR_dBs', SNR_dBs)
    return config