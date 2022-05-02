import torch
import numpy as np
import torch.nn as nn
import math

class NumTransmitterEncoding(nn.Module):

	def __init__(self, d_model, d_transmitter_encoding, max_transmitter, dropout=0.0):
		super(NumTransmitterEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.d_transmitter_encoding = d_transmitter_encoding

		NT = torch.zeros(int(max_transmitter), d_transmitter_encoding)
		num_transmitters = torch.arange(1.0, max_transmitter+1.0, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0.0, d_transmitter_encoding, 2).float() * (-math.log(np.float(2*max_transmitter)) / d_transmitter_encoding))
		NT[:, 0::2] = torch.sin(num_transmitters * div_term)
		NT[:, 1::2] = torch.cos(num_transmitters * div_term)
		NT = NT/math.sqrt(d_model)
		self.register_buffer('NT', NT)

	def forward(self, x):
		num_transmitter = x.shape[0]
		batch_size = x.shape[1]
		num_transmitter_encoding = self.NT[num_transmitter-1,:].expand(size=(num_transmitter, batch_size, self.d_transmitter_encoding))
		x = torch.cat((x, num_transmitter_encoding),dim=2)
		return self.dropout(x)