"""
ULA.py Unadjusted Langevin Algorithm

This class handle the ULA algorithm and the MMSE denoiser for the constellation set
"""

import numpy as np
import torch

from utils.util import *

class Unadjusted_langevin_algorithm():
    """
    ULA class to define the langevin algorithn plain (for each level)
    Input:
        generator: generator class, used to generate data
        n_samples: number of iterations to run the algorithm
        step: epsilon step size of the algorithm
    
    """
    def __init__(self, generator, n_samples, step, device):
        super(Unadjusted_langevin_algorithm, self).__init__()
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device

    
    def gaussian(self, zt, generator, sigma_annealed, NT, M):
        """
        Gaussian denoiser
        Input:
            generator: generator class, used to generate data
            sigma_annealed: sqrt of the variance of the noise annealed
            NT: number of users
            M: order of the constellation 
        Output:
            x_out: Estimation of the MMSE denoiser
        """

        #Calculate the distance of the estimated symbol to each true symbol from the constellation
        argr = torch.reshape(zt[:,0:NT],[-1,1]) - generator.QAM_const()[0].to(device=self.device)
        argi = torch.reshape(zt[:,NT:],[-1,1]) - generator.QAM_const()[1].to(device=self.device)

        #Reshape to handle batches
        argr = torch.reshape(argr, [-1, NT, M **2]) 
        argi = torch.reshape(argi, [-1, NT, M **2]) 

        #Softmax to calculate probabilites
        zt = torch.pow(argr,2) + torch.pow(argi,2)
        exp = -1.0 * (zt/(2.0 * sigma_annealed**2))
        exp = exp.softmax(dim=-1)

        #Multiplication of the numerator with each symbol
        xr = torch.mul(torch.reshape(exp,[-1,M **2]).float(), generator.QAM_const()[0].to(device=self.device))
        xi = torch.mul(torch.reshape(exp,[-1,M **2 ]).float(), generator.QAM_const()[1].to(device=self.device))

        #Summation and concatenation to obtain the real version of the complex symbol
        xr = torch.reshape(xr, [-1, NT, M **2]).sum(dim=-1)
        xi = torch.reshape(xi, [-1, NT, M **2]).sum(dim=-1)
        x_out = torch.cat((xr, xi), dim=-1)

        return x_out


    def forward(self, Z0, singulars, Sigma, Uh, Vh, y, noise_sigma, sigma_gaussian, sigma_L, batch_size, NT, M):
        """
        Forward pass
        Input:
            Z0: initial value
            singulars: vector with singular values
            Sigma: Matrix with the singular values
            y: observations
            Uh, Vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measuremnt noise
            sigma_gaussian:sqrt of the variance of the annealed noise at the level
            sigma_L: sqrt of the variance of the annealed noise at the last level 
            batch_size: number of channel samples
            NT: Number of users
            M: order of the modulation
        Output:
            Zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        Zi = Z0
        samples = []
        singulars = singulars.to(device=self.device)

        yT = batch_matvec_mul(Uh, y.float())
        ZT = batch_matvec_mul(Vh, Zi)
        A = torch.zeros((batch_size, 2 * NT)).to(device=self.device)

        #This step is to define which index value corresponds to noise_sigma > or < singualr * sigma_annealed
        index = noise_sigma.to(device=self.device) * torch.ones((batch_size,2 * NT)).to(device=self.device) < singulars.to(device=self.device) * sigma_gaussian

        #Position dependent step size
        A[index == True] = sigma_gaussian**2 - noise_sigma**2/singulars[index == True]**2 
        A[index == False] = sigma_gaussian**2 * (1 - singulars[index == False]**2 * (sigma_gaussian**2/noise_sigma**2))


        for i in range(self.n_samples):

            grad = torch.zeros((batch_size, 2 * NT)).to(device=self.device)

            #Score of the prior
            prior = (self.gaussian(Zi, self.generator, sigma_gaussian**2, NT, M) - Zi) / sigma_gaussian**2
            priorMul = batch_matvec_mul(Vh, prior)

            #Score of the likelihood
            diff =  (yT.float().to(device=self.device) - batch_matvec_mul(Sigma.float().to(device=self.device), ZT.float().to(device=self.device)))
            cov_diag = noise_sigma**2 * torch.ones( (batch_size, 2 * NT)).to(device=self.device) - sigma_gaussian**2 * (singulars**2).float().to(device=self.device)
            cov_diag[index == True] = -1 * cov_diag[index == True]
            cov = torch.diag_embed(cov_diag, dim1=1, dim2=2)
            cov_inv = torch.inverse(cov)
            aux = batch_matvec_mul(cov_inv, diff)
            grad_likelihood = batch_matvec_mul(torch.transpose(Sigma, 1, 2), aux)

            #Score of the posterior
            if torch.sum(index == False)!=0:
                grad[index == False] = grad_likelihood[index == False].to(device=self.device) +  priorMul[index == False].to(device=self.device)
            if torch.sum(index == True)!=0:
                grad[index == True] = grad_likelihood[index == True].to(device=self.device)

            #nosie defintiion
            noiseT = torch.randn(batch_size, 2 * NT).to(device=self.device)

            ZT = ZT + (self.step / sigma_L**2) * torch.mul(A, grad.to(device=self.device)) + np.sqrt( (2 * self.step) / sigma_L**2) * torch.mul(torch.sqrt(A), noiseT)

            Zi = batch_matvec_mul(torch.transpose(Vh, 1, 2), ZT)                                                                                           

            samples.append(Zi.cpu().detach().numpy())

        return Zi, samples

