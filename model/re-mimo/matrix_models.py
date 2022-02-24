import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
# from numpy import linalg as LA

##----ULA matrix-----##

def createCu(NT=6, NR=12):
    
    phi = np.zeros((NT,1))

    phi_mean_max = 10 + (NT/2 - 1) * 20
#     phi_mean_max = 20
    sigma = 10
    phi_mean = np.linspace(-phi_mean_max,phi_mean_max,NT)
    for uu in range(NT):
        phi[uu] = np.random.normal(phi_mean[uu], sigma, 1)

    j = complex(0,1)

    d = 1 / 2
    Cu = np.zeros((NR,NR, NT),dtype=complex)
    for uu in range(NT):
        for mm in range(NR):
            for nn in range(NR):
                Cu[mm,nn,uu] = np.exp(2 * np.pi * j * d * (mm-nn) * np.sin((phi_mean[uu] * np.pi)/180))\
                            * np.exp (-((sigma*np.pi)/180)**2/2 * (2 * np.pi * d * (mm-nn) * np.cos((phi_mean[uu] * np.pi)/180))**2 )

    return Cu

def createH(Cu, batch_size):
    HH = np.zeros((batch_size, NR, NT), dtype=complex)
    aux = np.zeros((batch_size,2 * NR ,2 * NT ))

    for ii in range(batch_size):
        for uu in range(NT):
            lamb, v = LA.eig(Cu[:,:,uu])
            e = np.random.multivariate_normal(np.zeros(2 * NR), 0.5*np.eye(2 * NR),1).view(np.complex128)[0]

            HH[ii,:,uu] = np.sqrt(1/NR) * np.dot(np.matmul(np.matmul(v,np.diag(np.sqrt(lamb))),np.transpose(np.conj(v))),e).ravel()

        h1 = np.concatenate([np.real(HH[ii,:,:]), -1. * np.imag(HH[ii,:,:])], axis=1)
        h2 = np.concatenate([np.imag(HH[ii,:,:]), np.real(HH[ii,:,:])], axis=1)
        aux[ii,:,:] = np.concatenate([h1, h2], axis=0)


    return np.float64(aux)

def createQR(Cu, batch_size, NT = 6, NR = 12):
    HH = np.zeros((batch_size, NR, NT), dtype=complex)
    aux = np.zeros((batch_size, 2 * NR, 2 * NR ))
    batchQ = np.zeros((batch_size, 2 * NT,2 * NR ))
    batchR = np.zeros((batch_size, 2 * NT,2 * NT ))
    QQ = np.zeros((NR, NT), dtype=complex)
    RR = np.zeros((NT, NT), dtype=complex)
    for ii in range(batch_size):
        for uu in range(NT):
            lamb, v = LA.eig(Cu[:,:,uu])
            e = np.random.multivariate_normal(np.zeros(2 * NR), 0.5*np.eye(2 * NR),1).view(np.complex128)[0]

            HH[ii,:,uu] = np.sqrt(1/NR) * np.dot(np.matmul(np.matmul(v,np.diag(np.sqrt(lamb))),np.transpose(np.conj(v))),e).ravel()

        QQ[:,:],RR[:,:] = np.linalg.qr(HH[ii,:,:])


        QQtr = np.transpose(np.conjugate(QQ)) 
        q1 = np.concatenate([np.real(QQtr), -1. * np.imag(QQtr)], axis=1)
        q2 = np.concatenate([np.imag(QQtr), np.real(QQtr)], axis=1)
        r1 = np.concatenate([np.real(RR), -1. * np.imag(RR)], axis=1)
        r2 = np.concatenate([np.imag(RR), np.real(RR)], axis=1)
        batchQ[ii,:,:] = np.concatenate([q1, q2], axis=0)
        batchR[ii,:,:] = np.concatenate([r1, r2], axis=0)

    return np.float64(batchQ), np.float64(batchR), np.float64(HH)

def createH_MoorePenrose(Cu, batch_size):
    H = np.zeros((batch_size, NR, NT), dtype=complex)
    invH = np.zeros((batch_size, NT, NR), dtype=complex)

    auxH = np.zeros((batch_size, NT, NT ))
    H_real = np.zeros((batch_size, 2 * NR, 2 * NT))
    
    auxHTH = np.zeros((batch_size, NT, NT ))
    H_tilde = np.zeros((batch_size, 2 * NT, 2 * NT))
    
    H_inv = np.zeros((batch_size, 2 * NT, 2 * NR))

    for ii in range(batch_size):
        for uu in range(NT):
            lamb, v = LA.eig(Cu[:,:,uu])
            e = np.random.multivariate_normal(np.zeros(2 * NR), 0.5*np.eye(2 * NR),1).view(np.complex128)[0]

            H[ii,:,uu] = np.dot(np.matmul(np.matmul(v,np.diag(np.sqrt(lamb))),np.transpose(np.conj(v))),e).ravel()
            
        invH[ii,:,:] = LA.pinv(H[ii,:,:])
        h1 = np.concatenate([np.real(H[ii,:,:]), -1. * np.imag(H[ii,:,:])], axis=1)
        h2 = np.concatenate([np.imag(H[ii,:,:]), np.real(H[ii,:,:])], axis=1)
        H_real[ii,:,:] = np.concatenate([h1, h2], axis=0)
        
        h1 = np.concatenate([np.real(invH[ii,:,:]), -1. * np.imag(invH[ii,:,:])], axis=1)
        h2 = np.concatenate([np.imag(invH[ii,:,:]), np.real(invH[ii,:,:])], axis=1)
        H_inv[ii,:,:] = np.concatenate([h1, h2], axis=0)
        
        auxHTH = np.matmul(invH[ii,:,:], H[ii,:,:])
        h1 = np.concatenate([np.real(auxHTH), -1. * np.imag(auxHTH)], axis=1)
        h2 = np.concatenate([np.imag(auxHTH), np.real(auxHTH)], axis=1)
        H_tilde[ii,:,:] = np.concatenate([h1, h2], axis=0)


    return np.double(H_real), np.double(H_inv), np.double(H_tilde)