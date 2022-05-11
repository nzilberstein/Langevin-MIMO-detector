import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

class MultiheadAttention(nn.Module):

	def __init__(self, embed_dim, num_heads, output_dim, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
		super(MultiheadAttention, self).__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if vdim is not None else embed_dim
		self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

		self.num_heads = num_heads
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

		self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

		if self._qkv_same_embed_dim is False:
			self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
			self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
			self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))

		if bias:
			self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
		else:
			self.register_parameter('in_proj_bias', None)
		self.out_proj = nn.Linear(embed_dim, output_dim, bias=bias)

		if add_bias_kv:
			self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
			self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
		else:
			self.bias_k = self.bias_v = None

		self.add_zero_attn = add_zero_attn

		self._reset_parameters()

	def _reset_parameters(self):
		if self._qkv_same_embed_dim:
			nn.init.xavier_uniform_(self.in_proj_weight)
		else:
			nn.init.xavier_uniform_(self.q_proj_weight)
			nn.init.xavier_uniform_(self.k_proj_weight)
			nn.init.xavier_uniform_(self.v_proj_weight)

		if self.in_proj_bias is not None:
			nn.init.constant_(self.in_proj_bias, 0.)
			nn.init.constant_(self.out_proj.bias, 0.)
		if self.bias_k is not None:
			nn.init.nn.init.xavier_normal_(self.bias_k)
		if self.bias_v is not None:
			nn.init.xavier_normal_(self.bias_v)

	def forward(self, query, key, value, key_padding_mask=None,
				need_weights=True, attn_mask=None):

		if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
			return F.multi_head_attention_forward(
				query, key, value, self.embed_dim, self.num_heads,
				self.in_proj_weight, self.in_proj_bias,
				self.bias_k, self.bias_v, self.add_zero_attn,
				self.dropout, self.out_proj.weight, self.out_proj.bias, 
				training=self.training,
				key_padding_mask=key_padding_mask, need_weights=need_weights, 
				attn_mask=attn_mask, use_separate_proj_weight=True,
				q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
				v_proj_weight=self.v_proj_weight)
		else:
			if not hasattr(self, '_qkv_same_embed_dim'):
				warnings.warn('A new version of MultiheadAttention module has been implemented. \
					Please re-train your model with the new module',
							  UserWarning)

			return F.multi_head_attention_forward(
				query, key, value, self.embed_dim, self.num_heads,
				self.in_proj_weight, self.in_proj_bias,
				self.bias_k, self.bias_v, self.add_zero_attn,
				self.dropout, self.out_proj.weight, self.out_proj.bias, 
				training=self.training,
				key_padding_mask=key_padding_mask, need_weights=need_weights, 
				attn_mask=attn_mask)