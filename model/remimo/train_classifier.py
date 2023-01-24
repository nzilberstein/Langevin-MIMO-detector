import torch
import numpy as np
import torch.nn as nn
import pickle
import os

from sample_generator import sample_generator
from iterative_classifier import iterative_classifier

# Parameters
NR = 64
NT_list = np.arange(16, 33)
NT_prob = NT_list/NT_list.sum()
mod_n = 16
d_transmitter_encoding = NR
d_model = 512
n_head = 8
nhid = d_model*4
nlayers = 16
dropout = 0.0

epoch_size = 5000
train_iter = 130*epoch_size

# Batch sizes for training and validation sets
train_batch_size = 256
mini_validtn_batch_size = 5000

learning_rate = 1e-4

corr_flag = True
batch_corr = True
rho_low = 0.55
rho_high = 0.75

validtn_NT_list = np.asarray([16, 32])
snrdb_list = {16:np.arange(11.0, 22.0), 32:np.arange(16.0, 27.0)}
factor_list = (validtn_NT_list/validtn_NT_list.sum())/snrdb_list[16].size

model_filename = './validtn_results/model.pth'
curr_accr = './validtn_results/curr_accr.txt'
load_pretrained_model = False
save_interim_model = True
save_to_file = True

def get_snr_range(NT):
    peak = NT*(5.0/16.0) + 6.0
    snr_low = peak
    snr_high = peak+10.0
    return (snr_low, snr_high)

def accuracy(out, j_indices):
    out = out.permute(1,2,0)
    out = out.argmax(dim=1)
    accuracy = (out == j_indices).sum().to(dtype=torch.float32)
    del out
    return accuracy/j_indices.numel()

def loss_function(criterion, out, j_indices):
    out = torch.cat(out, dim=1).permute(1,2,0)
    j_indices = j_indices.repeat(nlayers, 1)
    loss = criterion(out, j_indices)
    del out, j_indices
    return loss

def validate_model_given_data(model, validtn_H, validtn_y, validtn_j_indices, validtn_noise_sigma, device, criterion=None):
    with torch.no_grad():

        validtn_H = validtn_H.to(device=device)
        validtn_y = validtn_y.to(device=device)
        validtn_noise_sigma = validtn_noise_sigma.to(device=device)
        validtn_out = model.forward(validtn_H, validtn_y, validtn_noise_sigma)

        if (criterion):
            validtn_j_indices = validtn_j_indices.to(device=device)
            loss = loss_function(criterion, validtn_out, validtn_j_indices)
            validtn_j_indices = validtn_j_indices.to(device='cpu')

        validtn_out = validtn_out[-1].to(device='cpu')
        accr = accuracy(validtn_out, validtn_j_indices)

        del validtn_H, validtn_y, validtn_noise_sigma, validtn_out, validtn_j_indices

        if (criterion):
            return accr, loss.item()
        else:
            return accr, None

def mini_validation(model, mini_validation_dict, i, device, criterion=None, save_to_file=True):
    result_dict = {int(NT):{} for NT in validtn_NT_list}
    loss_list = []
    for index,NT in enumerate(validtn_NT_list):
        for snr in snrdb_list[NT]:
            big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = mini_validation_dict[NT][snr]
            accr, loss = validate_model_given_data(model, big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, device, criterion)
            result_dict[NT][snr] = accr
            loss_list.append(loss*factor_list[index])

    print('Validtn result, Accr for 16 : ', result_dict[16])
    print('Validation resut, Accr for 32 : ', result_dict[32])
    if (save_to_file):
        with open(curr_accr, 'w') as f:
            print((i, result_dict), file=f)
        print('Saved intermediate validation results at : ', curr_accr)

    if (criterion):
        return np.sum(loss_list)

def generate_big_validtn_data(generator, batch_size, corr_flag, rho, batch_corr, rho_low, rho_high):
    validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
    for NT in validtn_NT_list:
        for snr in snrdb_list[NT]:
            big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size, correlated_flag=corr_flag, rho=rho, batch_corr=batch_corr,rho_low=rho_low, rho_high=rho_high)
            validtn_data_dict[int(NT)][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
    return validtn_data_dict

def save_model_func(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_filename)
    print('******Model Saved********** at directory : ', model_filename)


def train(model, optimizer, lr_scheduler, generator , device='cpu'):

    mini_validation_dict = generate_big_validtn_data(generator, mini_validtn_batch_size, corr_flag, None, batch_corr, rho_low, rho_high)
    # Fix loss criterion
    criterion = nn.CrossEntropyLoss().to(device=device)
    model.train()
    epoch_count = 1

    for i in range(1, train_iter+1):

        # Randomly select number of transmitters
        NT = np.random.choice(NT_list, p=NT_prob)
        rho = np.random.triangular(rho_low, rho_high, rho_high)

        snr_low, snr_high = get_snr_range(NT)
        H, y, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=snr_low, snr_db_max=snr_high, batch_size=None, correlated_flag=corr_flag, rho=rho)

        H = H.to(device=device)
        y = y.to(device=device)
        noise_sigma = noise_sigma.to(device=device)

        out = model.forward(H,y, noise_sigma)

        del H, y, noise_sigma

        j_indices = j_indices.to(device=device)
        loss = loss_function(criterion, out, j_indices)
        del j_indices, out
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        del loss

        if (i%epoch_size==0):
            print('iteration number : ', i, 'Epoch : ', epoch_count, 'User : ', NT, 'loss : ', loss_item)
            print('Now validating')

            model.eval()
            mini_validtn_loss = mini_validation(model, mini_validation_dict, i, device, criterion, save_to_file)
            print('Mini validation loss : ', mini_validtn_loss)
            lr_scheduler.step(mini_validtn_loss)

            model.train()
            if (save_interim_model):
                save_model_func(model, optimizer)

            epoch_count = epoch_count+1

def main():
    generator = sample_generator(train_batch_size, mod_n, NR)
    device = 'cuda'
    model = iterative_classifier(d_model, n_head, nhid, nlayers, mod_n, NR, d_transmitter_encoding, generator.real_QAM_const, generator.imag_QAM_const, generator.constellation, device, dropout)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    if (load_pretrained_model):
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08, verbose = True)
        print('*******Successfully loaded pre-trained model***********')
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08, verbose = True)

    train(model, optimizer, lr_scheduler, generator, device)
    print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
    main()
