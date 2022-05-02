"""
langevin.py Langevin class to handle the annealed process
"""

import torch
from model.unadjusted_langevin import *
from model.MLP import *
import torch.nn as nn

class Langevin():
    """
    ULA class to define the langevin algorithn plain (for each level)
    Input:
        sigma_annealead_noise: vector with all the sqrt variances of the noise
        generator: generator class, used to generate data
        n_samples: number of iterations to run the algorithm
        step: epsilon step size of the algorithm
    
    """
    def __init__(self, sigma_annealead_noise, generator, n_samples, step, device):
        super(Langevin, self).__init__()
        self.num_noise_levels = sigma_annealead_noise.shape[0]
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device
        self.sigma_gaussian = sigma_annealead_noise
        self.Langevin_base = Unadjusted_langevin_algorithm(self.generator, self.n_samples, self.step, self.device)

    def forward(self, singulars, Sigma, Uh, Vh, y, noise_sigma, NT, M, Temp):
        """
        Forward pass
        Input:
            singulars: vector with singular values
            Sigma: Matrix with the singular values
            y: observations
            Uh, Vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measuremnt noisea
            NT: Number of users
            M: order of the modulation
        Output:
            Zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        r1 = 1
        r2 = -1
        bs = Sigma.shape[0]
        Z_init = ((r1 - r2) * torch.rand(bs, 2 * NT) + r2).to(device=self.device)
        sample_list = []

        for index in range(self.num_noise_levels):
            Zi, samples = self.Langevin_base.forward(Z_init, singulars, Sigma, Uh, Vh, y, noise_sigma, self.sigma_gaussian[index], self.sigma_gaussian[-1], bs, NT, M, Temp)
            sample_list.append(samples)

            #Define the initial value of the next level
            Z_init = torch.clone(Zi).to(device=self.device)
        return Zi, sample_list
    
    
class Langevin_two_modulations():
    """
    ULA class to define the langevin algorithm plain for two modulations (for each level)
    Input:
        sigma_annealead_noise: vector with all the sqrt variances of the noise
        generator: generator class, used to generate data
        n_samples: number of iterations to run the algorithm
        step: epsilon step size of the algorithm
    
    """
    def __init__(self, sigma_annealead_noise, generator, n_samples, step, device):
        super(Langevin_two_modulations, self).__init__()
        self.num_noise_levels = sigma_annealead_noise.shape[0]
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device
        self.sigma_gaussian = sigma_annealead_noise
        self.Langevin_base = Unadjusted_langevin_algorithm_two_modulations(self.generator, self.n_samples, self.step, self.device)

    def forward(self, singulars, Sigma, Uh, Vh, y, noise_sigma, NT, M1, M2, Temp):
        """
        Forward pass
        Input:
            singulars: vector with singular values
            Sigma: Matrix with the singular values
            y: observations
            Uh, Vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measuremnt noisea
            NT: Number of users
            M: order of the modulation
        Output:
            Zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        r1 = 1
        r2 = -1
        bs = Sigma.shape[0]
        Z_init = ((r1 - r2) * torch.rand(bs, 2 * NT) + r2).to(device=self.device)
        sample_list = []

        for index in range(self.num_noise_levels):
            Zi, samples = self.Langevin_base.forward(Z_init, singulars, Sigma, Uh, Vh, y, noise_sigma, self.sigma_gaussian[index], self.sigma_gaussian[-1], bs, NT, M1, M2, Temp)
            sample_list.append(samples)

            #Define the initial value of the next level
            Z_init = torch.clone(Zi).to(device=self.device)
        return Zi, sample_list
    
    
class Langevin_unfolded(nn.Module):
    """
    ULA class to define the langevin algorithm plain for unolded langevin (for each level)
    Input:
        sigma_annealead_noise: vector with all the sqrt variances of the noise
        generator: generator class, used to generate data
        n_samples: number of iterations to run the algorithm
        step: epsilon step size of the algorithm
    
    """
    def __init__(self, NT, NR, sigma_annealead_noise, generator, n_samples, step, device):
        super(Langevin_unfolded, self).__init__()
        self.num_noise_levels = sigma_annealead_noise.shape[0]
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device
        self.sigma_gaussian = sigma_annealead_noise
        self.Langevin_base = nn.ModuleList([Unadjusted_langevin_algorithm_unfolded(NT, NR, self.generator, self.n_samples, self.step, self.device) for i in range(self.num_noise_levels)])

    def forward(self, singulars, Sigma, Uh, Vh, y, noise_sigma, NT, M, Temp):
        """
        Forward pass
        Input:
            singulars: vector with singular values
            Sigma: Matrix with the singular values
            y: observations
            Uh, Vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measuremnt noisea
            NT: Number of users
            M: order of the modulation
        Output:
            Zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        r1 = 1
        r2 = -1
        bs = Sigma.shape[0]
        Z_init = ((r1 - r2) * torch.rand(bs, 2 * NT) + r2).to(device=self.device)
        sample_list = []

        for index, langevin_base in enumerate(self.Langevin_base):
            Zi, samples = langevin_base.forward(Z_init, singulars, Sigma, Uh, Vh, y, noise_sigma, self.sigma_gaussian[index], self.sigma_gaussian[-1], bs, NT, M, Temp)
            sample_list.append(Zi)

            #Define the initial value of the next level
            Z_init = torch.clone(Zi).to(device=self.device)
        return Zi, sample_list[1:]