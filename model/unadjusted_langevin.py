"""
ULA.py Unadjusted Langevin Algorithm

This class handle the ULA algorithm and the MMSE denoiser for the constellation set
"""

import numpy as np
import torch

from utils.util import *

class Unadjusted_langevin_algorithm():
    def __init__(self, generator, n_samples, step, device):
        super(Unadjusted_langevin_algorithm, self).__init__()
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device

    
    def gaussian(self, zt, generator, noise_sigma, NT, M):
        argr = torch.reshape(zt[:,0:NT],[-1,1]) - generator.QAM_const()[0].to(device=self.device)
        argi = torch.reshape(zt[:,NT:],[-1,1]) - generator.QAM_const()[1].to(device=self.device)

        argr = torch.reshape(argr, [-1, NT, M **2]) 
        argi = torch.reshape(argi, [-1, NT, M **2]) 

        zt = torch.pow(argr,2) + torch.pow(argi,2)
        exp = -1.0 * (zt/(2.0 * noise_sigma**2))
        exp = exp.softmax(dim=-1)

        xr = torch.mul(torch.reshape(exp,[-1,M **2]).float(), generator.QAM_const()[0].to(device=self.device))
        xi = torch.mul(torch.reshape(exp,[-1,M **2 ]).float(), generator.QAM_const()[1].to(device=self.device))

        xr = torch.reshape(xr, [-1, NT, M **2]).sum(dim=-1)
        xi = torch.reshape(xi, [-1, NT, M **2]).sum(dim=-1)
        x_out = torch.cat((xr, xi), dim=-1)

        return x_out


    def forward(self, Z0, singulars, H, Uh, Vh, y, noise_sigma, sigma_gaussian, sigma_L, batch_size, NT, M):

        Zi = Z0
        samples = []
        yT = batch_matvec_mul(Uh, y.float())
        ZT = batch_matvec_mul(Vh, Zi)
        singulars = singulars.to(device=self.device)
        A = torch.zeros((batch_size, 2 * NT)).to(device=self.device)


        index = noise_sigma.to(device=self.device) * torch.ones((batch_size,2 * NT)).to(device=self.device) < singulars.to(device=self.device) * sigma_gaussian

        A[index == True] = sigma_gaussian**2 - noise_sigma**2/singulars[index == True]**2 
    #     A[index == False] = -(sigma_gaussian**2 - sigma**2/singulars[index == False]**2)
        A[index == False] = sigma_gaussian**2 * (1 - singulars[index == False]**2 * (sigma_gaussian**2/noise_sigma**2))


        for i in range(self.n_samples):

            grad = torch.zeros((batch_size, 2 * NT)).to(device=self.device)
            prior = (self.gaussian(Zi, self.generator, sigma_gaussian**2, NT, M) - Zi) / sigma_gaussian**2
            priorMul = batch_matvec_mul(Vh, prior)

            diff =  (yT.float().to(device=self.device) - batch_matvec_mul(H.float().to(device=self.device), ZT.float().to(device=self.device)))
            cov_diag = noise_sigma**2 * torch.ones( (batch_size, 2 * NT)).to(device=self.device) - sigma_gaussian**2 * (singulars**2).float().to(device=self.device)
            cov_diag[index == True] = -1 * cov_diag[index == True]
            cov = torch.diag_embed(cov_diag, dim1=1, dim2=2)
            cov_inv = torch.inverse(cov)
            aux = batch_matvec_mul(cov_inv, diff)
            grad_likelihood = batch_matvec_mul(torch.transpose(H, 1, 2), aux)

            if torch.sum(index == False)!=0:
                grad[index == False] = grad_likelihood[index == False].to(device=self.device) +  priorMul[index == False].to(device=self.device)
            if torch.sum(index == True)!=0:
                grad[index == True] = grad_likelihood[index == True].to(device=self.device)


            noiseT = torch.randn(batch_size, 2 * NT).to(device=self.device)

            ZT = ZT + (self.step / sigma_L**2) * torch.mul(A, grad.to(device=self.device)) + np.sqrt( (2 * self.step) / sigma_L**2) * torch.mul(torch.sqrt(A), noiseT)

            Zi = batch_matvec_mul(torch.transpose(Vh, 1, 2), ZT)                                                                                           

            samples.append(Zi.cpu().detach().numpy())

        return Zi, samples

