import numpy as np
import math
import torch
import torch.nn as nn

class TransformerDecoderLayer(nn.Module):

	def __init__(self, d_model, NR, mod_n):
		super(TransformerDecoderLayer, self).__init__()
		# Implementation of Feedforward model
		qam_const = int(np.sqrt(mod_n))
		initial_dim = d_model + 4*NR + mod_n + 1
		interim_dim_1 = (initial_dim+1)//2
		interim_dim_2 = (interim_dim_1+1)//2
		final_dim = mod_n

		self.linear1 = nn.Linear(initial_dim, interim_dim_1)
		self.linear2 = nn.Linear(interim_dim_1, interim_dim_2)
		self.linear3 = nn.Linear(interim_dim_2, final_dim)
		self.activation_1 = nn.ReLU()
		self.activation_2 = nn.ReLU()
		self.mod_n = mod_n

	def gen_decoder_input(self, st, common_input, noise_sigma, NT):
		noise_sigma_normalized = ((noise_sigma)/np.sqrt(2.0*NT)).expand(NT, -1).unsqueeze(dim=-1)
		decoder_embed = torch.cat((st, common_input, noise_sigma_normalized), dim=-1)
		del noise_sigma_normalized
		return decoder_embed

	def forward(self, st, common_input, noise_sigma, NT):
		decoder_embed = self.gen_decoder_input(st, common_input, noise_sigma, NT)
		del st, common_input, noise_sigma
		out = self.linear1(decoder_embed)
		out = self.linear2(self.activation_1(out))
		out = self.linear3(self.activation_2(out))
		del decoder_embed
		return out
