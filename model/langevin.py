"""
langevin.py Langevin class to handle the annealed process
"""

import torch
from model.unadjusted_langevin import *


class Langevin():

    def __init__(self, sigma_annealead_noise, generator, n_samples, step, device):
        super(Langevin, self).__init__()
        self.num_noise_levels = sigma_annealead_noise.shape[0]
        self.generator = generator
        self.n_samples = n_samples
        self.step = step
        self.device = device
        self.sigma_gaussian = sigma_annealead_noise
        self.Langevin_base = Unadjusted_langevin_algorithm(self.generator, self.n_samples, self.step, self.device)

    def forward(self, singulars, Sigma, Uh, Vh, y, sigma, bs, NT, M):
        r1 = 1
        r2 = -1
        Z_init = ((r1 - r2) * torch.rand(bs, 2 * NT) + r2).to(device=self.device)
        sample_list = []

        for index in range(self.num_noise_levels):
            Zi, samples = self.Langevin_base.forward(Z_init, singulars, Sigma, Uh, Vh, y, sigma, self.sigma_gaussian[index], self.sigma_gaussian[-1], bs, NT, M)
            sample_list.append(samples)
            Z_init = torch.clone(Zi).to(device=self.device)
#             print(index, Zi, grad,prior)
        return Zi, sample_list