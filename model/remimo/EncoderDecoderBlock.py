import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerEncoder import TransformerEncoder
from TransformerEncoderLayer import TransformerEncoderLayer
from TransformerDecoderLayer import TransformerDecoderLayer

def frobenius_norm(matrix, matrix_transpose=None):
    if (not matrix_transpose):
        matrix_transpose = matrix.permute(0,2,1)
    product = torch.matmul(matrix, matrix_transpose)
    return torch.einsum(('ijj->i'),(product))

class EncoderDecoderBlock(nn.Module):

    def __init__(self, d_model, nhead, NR, mod_n, real_QAM_const, imag_QAM_const, constel, device, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderDecoderBlock, self).__init__()

        encoder_input_dim = 4*NR + mod_n + d_model

        encoder_layer = TransformerEncoderLayer(d_model, nhead, encoder_input_dim, mod_n, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, 1)
        self.decoder = TransformerDecoderLayer(d_model, NR, mod_n)
        self.theta = nn.Parameter(torch.Tensor([1.0]))
        nn.init.normal_(self.theta, mean=1.0, std=0.1)
        self.real_QAM_const = real_QAM_const.to(device=device)
        self.imag_QAM_const = imag_QAM_const.to(device=device)
        self.constel = constel.to(device=device)
        self.NR = NR
        self.device = device
        self.mod_n = mod_n
        self.constel_size = np.int(np.sqrt(mod_n))
        self.nhead = nhead

    def gen_common_input(self, xt, H, y, noise_sigma, NT):

        xt_probs = xt.softmax(dim=-1)

        x_real = (xt_probs *self.real_QAM_const).sum(dim=-1)
        x_imag = (xt_probs *self.imag_QAM_const).sum(dim=-1)

        xt_val = torch.cat((x_real, x_imag), dim=0).permute(1, 0)
        delta_y = y - torch.einsum(('ijk,ik->ij'), (H, xt_val))

        del y, xt_probs, xt_val, x_imag, x_real

        tgt = torch.chunk(H, 2, dim=2)[0].permute(2,0,1)
        # Normalizing y
        
        min_value = 0.3
        attn_weights = torch.zeros((self.nhead * H.shape[0], NT, NT))
        Hcomplex = H[:,0:self.NR,0:NT] +1j*H[:,self.NR:,0:NT]
        
        corr = torch.zeros((self.nhead * H.shape[0], NT, NT))

        del H

        delta_y = delta_y/np.sqrt(2.0*NT)
        delta_y = torch.unsqueeze(delta_y, dim=0).expand(NT,-1,-1)

        final_repr_encoder = torch.cat((delta_y, tgt, xt), dim=-1)
        final_repr_decoder = torch.cat((delta_y, tgt, xt), dim=-1)

        del delta_y, tgt, xt
        return final_repr_encoder, final_repr_decoder, attn_weights.to(device=self.device)

    def forward(self, st, xt, H, y, noise_sigma, NT, index, attn_weights = None, save_attn_weight = False):

        encoder_input, decoder_input, attn_weights = self.gen_common_input(xt, H, y, noise_sigma, NT)
        del xt, H, y
        encoder_out = self.encoder.forward(st, encoder_input, index, save_attn_weight, mask = attn_weights)
        del encoder_input, st

        decoder_out = self.decoder.forward(encoder_out, decoder_input, noise_sigma, NT)
        del decoder_input, noise_sigma
        return encoder_out, decoder_out