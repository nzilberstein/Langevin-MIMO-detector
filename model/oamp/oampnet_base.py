import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def frobenius_norm(matrix, matrix_transpose=None):
    if (not matrix_transpose):
        matrix_transpose = matrix.permute(0,2,1)
    product = torch.matmul(matrix, matrix_transpose)
    return torch.einsum(('ijj->i'),(product))


class oampnet_base(nn.Module):

    def __init__(self, constel, real_QAM_const, imag_QAM_const, device):
        super(oampnet_base, self).__init__()

        self.theta = nn.Parameter(torch.tensor([1.0]))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.device = device
        self.real_QAM_const = real_QAM_const.to(device=device)
        self.imag_QAM_const = imag_QAM_const.to(device=device)
        self.constel = constel.to(device=device)
        self.constel_size = self.constel.numel()


    def get_v2t(self, H, y, x_out, noise_sigma):
        NR = y.shape[1]/2.0
        H_t = H.permute(0,2,1)
        HTH = torch.matmul(H_t, H)
        HHT = torch.matmul(H, H_t)
        h_frobenius = frobenius_norm(H_t)
        rt = y.double() - torch.einsum(('ijk,ik->ij'), (H.double(), x_out.double()))
        v2t = (torch.pow(rt,2).sum(dim=-1) - (NR)*(noise_sigma.to(device=self.device)**2))/(h_frobenius)
        v2t = torch.max(v2t, torch.Tensor([1e-9]).to(device=self.device))
        return v2t, HHT, H_t, rt, NR

    def get_v2t_wt(self, H, y, x_out, noise_sigma):
        NT = H.shape[-1]/2.0
        v2t, HHT, H_t , rt, NR = self.get_v2t(H, y, x_out, noise_sigma)
        lam = torch.eye(n=int(2*NR)).expand(size=HHT.shape).to(device=self.device)
        lam = ((noise_sigma.to(device=self.device)**2)/2.0).view(-1,1,1)*lam
        inv_term = torch.inverse(v2t.view(-1,1,1)*HHT + lam)
        interim = torch.matmul(H_t.double(), inv_term.double())
        what = v2t.view(-1,1,1)*interim
        wt = ((2.0*NT)*(what.double()))/(torch.einsum(('ijj->i'),(torch.matmul(what.double(), H.double()))).view(-1,1,1))
        del lam, inv_term, what
        return v2t, wt, rt, NT

    def get_tau(self, H, y, x_out, noise_sigma):
        v2t, wt, rt, NT = self.get_v2t_wt(H, y, x_out, noise_sigma)
        wt = wt.double()*self.gamma.double()
        wt_H = torch.matmul(wt, H.double())
        c_t = torch.eye(n=wt_H.shape[1]).expand(size=wt_H.shape).to(device=self.device) - self.theta*wt_H
        tau2t = (frobenius_norm(c_t)*v2t + ((self.theta**2)*(noise_sigma.to(device=self.device)**2)*(frobenius_norm(wt)/2.)))/(2.0*np.float(NT))
        zt = x_out + (torch.einsum(('ijk,ik->ij'), (wt, rt)))
        del v2t, wt, rt, NT, wt_H, c_t
        return tau2t, zt

    def process_forward(self, H, y, x_out, noise_sigma):

        tau2t, zt = self.get_tau(H, y, x_out, noise_sigma)
        zt = zt.unsqueeze(dim=-1).expand(-1,-1, self.constel_size)
        zt = torch.pow(zt - self.constel, 2)
        zt = -1.0 * (zt/(2.0*tau2t.view(-1,1,1)))
        zt_probs = zt.softmax(dim=-1)
        del zt

        x_out = (zt_probs* self.constel).sum(dim=-1)
        del zt_probs

        return x_out

    def forward(self, H, y, x_out, noise_sigma):
        return self.process_forward(H, y, x_out, noise_sigma)