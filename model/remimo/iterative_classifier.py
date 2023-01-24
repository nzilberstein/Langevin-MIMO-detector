import torch
import numpy as np
import torch.nn as nn
import math

from EncoderDecoderBlock import EncoderDecoderBlock
from NumTransmitterEncoding import NumTransmitterEncoding

class iterative_classifier(nn.Module):

    def __init__(self, d_model, n_head, nhid, nlayers, mod_n, NR, d_transmitter_encoding, real_QAM_const, imag_QAM_const, constel, device, dropout=0.0):
        super(iterative_classifier, self).__init__()
        self.d_model = d_model
        self.mod_n = mod_n
        self.device = device
        self.constel = constel.to(device=device)
        self.constel_size = constel.numel()
        self.NR = NR

        # source embeddings
        initial_dim = 4*NR + d_transmitter_encoding + 1
        interim_dim = d_model*4

        self.encoder_embed = nn.Sequential(nn.Linear(initial_dim, interim_dim), nn.ReLU(), nn.Linear(interim_dim, d_model))

        # Iterative Encoding-Decoding Blocks 
        self.iterative_blocks = nn.ModuleList([EncoderDecoderBlock(d_model, n_head, NR, mod_n, real_QAM_const, imag_QAM_const, constel, device, nhid, dropout) for i in range(nlayers)])

        # Num Transmitter encoder
        self.num_transmitter_encoder = NumTransmitterEncoding(d_model, d_transmitter_encoding, max_transmitter=NR, dropout=dropout)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_input(self, H, y, noise_sigma, NT):
        tgt = torch.chunk(H, 2, dim=2)[0].permute(2,0,1)
        # Normalizing y
        y = y/np.sqrt(2.0*NT)
        y = torch.unsqueeze(y, dim=0).expand(NT,-1,-1)
        noise_sigma_normalized = ((noise_sigma)/np.sqrt(2.0*NT)).expand(NT, -1).unsqueeze(dim=-1)
        src = torch.cat((y, tgt, noise_sigma_normalized), dim=-1)
        del H,y, noise_sigma, tgt
        return src

    def forward(self, H, y, noise_sigma, attn_mask = None, save_attn_weight=False):
        NT = H.shape[-1]//2
        sout = self.generate_input(H,y, noise_sigma, NT)
        sout = self.num_transmitter_encoder(sout)
        sout = self.encoder_embed(sout) * math.sqrt(self.d_model)
        xout = torch.zeros(NT, H.shape[0], self.mod_n)
        sout = sout.to(device=self.device)
        xout = xout.to(device=self.device)

        x_list = []

        for index, encoder_decoder in enumerate(self.iterative_blocks):
            sout, xout = encoder_decoder.forward(sout, xout, H, y, noise_sigma, NT, index, attn_mask, save_attn_weight)
            x_list.append(xout)

        return x_list