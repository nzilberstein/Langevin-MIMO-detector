import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.nn.modules.normalization import LayerNorm
from MultiheadAttention import MultiheadAttention

attn_weight_filename = './validtn_results/attn_weights/'
attn_weight_format = '_attn_weight.pickle'

def save_attention_weight(index, attn_output_weights):
    NT = attn_output_weights.shape[-1]
    filename = attn_weight_filename + str(NT) + '_' + str(index) + attn_weight_format
    with open(filename, 'wb') as handle:
        pickle.dump(attn_output_weights.detach().to(device='cpu').numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, encoder_input_dim, mod_n, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(encoder_input_dim, nhead, d_model, dropout=dropout)
        # Implementation of Feedforward model
        initial_dim = d_model
        self.linear1 = nn.Linear(initial_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, common_input, index, save_attn_weight, src_mask=None, src_key_padding_mask=None):
        src_concat = torch.cat((src, common_input), dim=-1)
        del common_input

        if (save_attn_weight):
            src2, attn_output_weights = self.self_attn(src_concat, src_concat, src_concat, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            save_attention_weight(index, attn_output_weights)
            del attn_output_weights
        else:
            src2 = self.self_attn(src_concat, src_concat, src_concat, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        del src_concat
        src = src + self.dropout1(src2)
        del src2
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        del src2
        src = self.norm2(src)
        return src
