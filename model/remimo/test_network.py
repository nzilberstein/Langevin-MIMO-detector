import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F

from collections import defaultdict
from sample_generator import sample_generator
from iterative_classifier import iterative_classifier

# Parameters
NR = 64
mod_n = 16
d_transmitter_encoding = NR
d_model = 512
n_head = 8
nhid = d_model*4
nlayers = 16
dropout = 0.0

# Batch sizes for training and validation sets
validtn_batch_size = 5000
validtn_iter = 2000

M = int(np.sqrt(mod_n))
sigConst = np.linspace(-M+1, M-1, M) 
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

validtn_NT_list = np.asarray([16, 32])
snrdb_list = {16:np.arange(11.0, 19.0), 32:np.arange(15.0, 24.0)}
corr_list = np.asarray([0.60, 0.70])

corr_flag = True

validtn_filename = './final_results/network_fullcorr_validtn_results.pickle'
model_filename = './validtn_results/model.pth'

def accuracy(out, j_indices):
	out = out.permute(1,2,0)
	out = out.argmax(dim=1)
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

def bit_indices(indices, mod_n):
	real_indices = (indices//np.sqrt(mod_n)).to(dtype=torch.int32)
	imag_indices = (indices%np.sqrt(mod_n)).to(dtype=torch.int32)
	joint_bit_indices = torch.cat((real_indices, imag_indices), dim=-1)
	return joint_bit_indices

def sym_accuracy(out, j_indices):
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

def bit_accuracy(out, j_indices):
	out = out.permute(1,2,0)
	out = out.argmax(dim=1)
	bit_out_indices = bit_indices(out, mod_n)
	bit_j_indices = bit_indices(j_indices, mod_n)
	return sym_accuracy(bit_out_indices, bit_j_indices)

def validate_model_given_data(model, validtn_H, validtn_y, validtn_j_indices, validtn_noise_sigma, device):
	with torch.no_grad():

		validtn_H = validtn_H.to(device=device)
		validtn_y = validtn_y.to(device=device)
		validtn_noise_sigma = validtn_noise_sigma.to(device=device)
		validtn_out = model.forward(validtn_H, validtn_y, validtn_noise_sigma)

		validtn_out = validtn_out[-1].to(device='cpu')
		accr = accuracy(validtn_out, validtn_j_indices)

		del validtn_H, validtn_y, validtn_out, validtn_noise_sigma

	return accr


def validate_model(model, generator, device, save_result=True):
	result_dict = {int(NT):{rho:defaultdict(float) for rho in corr_list} for NT in validtn_NT_list}
	for iter in range(validtn_iter):
		validtn_data_dict = generate_big_validtn_data(generator, validtn_batch_size)
		for NT in validtn_NT_list:
			for rho in corr_list:
				for snr in snrdb_list[NT]:
					big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = validtn_data_dict[NT][rho][snr]
					accr = validate_model_given_data(model, big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, device)
					result_dict[NT][rho][snr] =  result_dict[NT][rho][snr] + (accr-result_dict[NT][rho][snr])/float(iter+1.0)

		if (save_result):
			with open(validtn_filename, 'wb') as handle:
				pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('Intermediate Test results saved at : ', validtn_filename)
		print('Big Validtn result, Accr for 16 : ', result_dict[16])
		print('Big Validation resut, Accr for 32 : ', result_dict[32])


def generate_big_validtn_data(generator, batch_size):
	validtn_data_dict = {int(NT):{rho:{} for rho in corr_list} for NT in validtn_NT_list}
	for NT in validtn_NT_list:
		for rho in corr_list:
			for snr in snrdb_list[NT]:
				big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size, correlated_flag=corr_flag, rho=rho)
				validtn_data_dict[int(NT)][rho][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
	return validtn_data_dict

def test(model, generator, device):
	model.eval()

	# Testing Trained Network
	validate_model(model, generator, device, True)

def main():
	generator = sample_generator(validtn_batch_size, mod_n, NR)
	device = 'cuda'
	model = iterative_classifier(d_model, n_head, nhid, nlayers, mod_n, NR, d_transmitter_encoding, generator.real_QAM_const, generator.imag_QAM_const, generator.constellation, device, dropout)
	model = model.to(device=device)

	checkpoint = torch.load(model_filename)
	model.load_state_dict(checkpoint['model_state_dict'])
	print('*******Successfully loaded pre-trained model*********** from directory : ', model_filename)

	test(model, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
