import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F
# import cvxpy as cp
# from gurobipy import *

def mmse(y, H, noise_sigma, device):
    two_NT = int(H.shape[-1])
    H_t = H.permute(0,2,1)
    Hty = torch.einsum(('ijk,ik->ij'), (H_t, y))
    HTH = torch.matmul(H_t, H).to(device=device)
    HtHinv = torch.inverse(HTH + ((torch.pow(noise_sigma, 2)/2.0).view(-1, 1, 1))*torch.eye(n=two_NT).expand(size=HTH.shape).to(device=device))
    x = torch.einsum(('ijk,ik->ij'), (HtHinv, Hty))
    return x

def sym_detection(x_hat, real_QAM_const, imag_QAM_const):
    x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
    x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
    x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

    x_real = torch.pow(x_real - real_QAM_const, 2)
    x_imag = torch.pow(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    x_indices = torch.argmin(x_dist, dim=-1)
    return x_indices

def ampSolver(hBatch, yBatch, Symb, noise_sigma):
    def F(x_in, tau_l, Symb):
        arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
        exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
        prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
        f = np.matmul(prob, Symb.reshape((1,-1,1)))
        return f

    def G(x_in, tau_l, Symb):
        arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
        exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
        prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
        g = np.matmul(prob, Symb.reshape((1,-1,1)) ** 2) - F(x_in, tau_l, Symb) ** 2
        return g

    numIterations = 50
    NT = hBatch.shape[2]
    NR = hBatch.shape[1]
    N0 = noise_sigma ** 2 / 2.
    xhat = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
    z = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
    r = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[1], 1))
    tau = np.zeros((numIterations, hBatch.shape[0], 1, 1))
    r[0] = yBatch
    for l in range(numIterations-1):
        z[l] = xhat[l] + np.matmul(hBatch.transpose((0,2,1)), r[l])
        xhat[l+1] = F(z[l], N0 * (1.+tau[l]), Symb)
        tau[l+1] = float(NT) / NR / N0 * np.mean(G(z[l], N0 * (1. + tau[l]), Symb), axis=1, keepdims=True)
        r[l+1] = yBatch - np.matmul(hBatch, xhat[l+1]) + tau[l+1]/(1.+tau[l]) * r[l]

    return xhat[l+1]

def sdrSolver(hBatch, yBatch, constellation, NT):
    results = []
    for i, H in enumerate(hBatch):
        y = yBatch[i]
        s = cp.Variable((2*NT,1))
        S = cp.Variable((2*NT, 2*NT))
        objective = cp.Minimize(cp.trace(H.T @ H @ S) - 2. * y.T @ H @ s)
        constraints = [S[i,i] <= (constellation**2).max() for i in range(2*NT)]
        constraints += [S[i,i] >= (constellation**2).min() for i in range(2*NT)]
        constraints.append(cp.vstack([cp.hstack([S,s]), cp.hstack([s.T,[[1]]])]) >> 0)
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        results.append(s.value)
    results = np.array(results)
    return results

def zf_detector(y, H):
    return np.matmul(np.linalg.pinv(H), y)

def symbol_detection(y, constellation):
    return np.expand_dims(np.argmin(np.abs(y-np.expand_dims(constellation, 0)), axis=1),1)


def pic_detector(y, H, first_stage, constellation):
    #First detection stage
    x_1st = first_stage(y, H)
    x_1st_indices = symbol_detection(x_1st, constellation)
    x_1st = constellation[x_1st_indices]

    #PIC detection
    x_pic = np.zeros_like(x_1st)
    for k in range(0,x_pic.shape[0]):
        x_1st_k = np.copy(x_1st)
        x_1st_k[k, 0] = 0
        y_k = y - np.matmul(H, x_1st_k)
        H_k = np.linalg.pinv(np.expand_dims(H[:,k], 1))
        x_pic[k,0] = np.matmul(H_k, y_k)  
    return x_pic

def blast_eval(yBatch, hBatch, sigConst, NT, NR):
    batch_size = hBatch.shape[0]
    shatBatch = np.zeros([batch_size, 2*NT])
    for i, h in enumerate(hBatch):
        shatBatch[i] = np.reshape(pic_detector(yBatch[i], hBatch[i], zf_detector, sigConst), (-1))

    return shatBatch

def mlSolver(hBatch, yBatch, Symb):
    results = []
    status = []
    m = len(hBatch[0,0,:])
    n = len(hBatch[0,:,0])
#     print(m, n)
    for idx_, Y in enumerate(yBatch):
#         if idx_ % 10 == 0:
#             print(idx_ / float(len(yBatch)) * 100. ,"% completed")

        H = hBatch[idx_]
        model = Model('mimo')
        k = len(Symb)
        Z = model.addVars(m, k, vtype=GRB.BINARY, name='Z')
        S = model.addVars(m, ub=max(Symb)+.1, lb=min(Symb)-0.1,  name='S')
        E = model.addVars(n, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='E')
        model.update() 

        # Defining S[i]
        for i in range(m):
            model.addConstr(S[i] == quicksum(Z[i,j] * Symb[j] for j in range(k)))
            #S[i] == quicksum(Z[i,j] * Symb[j] for j in range(k))

        # Constraint on Z[i,j]
        model.addConstrs((Z.sum(j,'*') == 1
                         for j in range(m)), name='Const1')


        # Defining E
        for i in range(n):
            E[i] = quicksum(H[i][j] * S[j] for j in range(m)) - Y[i][0]
        #for i in range(m):
        #    E[i] = quicksum(quicksum(H[l,i]*H[l,j] for l in range(n)) * S[j] 
        #            - quicksum(H[ll,i]*Y[ll] for ll in range(n))  
        #            for j in range(m)) 

        # Defining the objective function
        obj = E.prod(E)  
        model.setObjective(obj, GRB.MINIMIZE)
        model.Params.logToConsole = 0
        model.setParam('TimeLimit', 100)
        model.update()

        model.optimize()

        #model.write('MIMO_BPSK.txt')

        #print('')
        #print('Solution:')
        #print('')

        # Retrieve optimization result
        solution = model.getAttr('X', S)
        status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
#         print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
#         if model.getAttr(GRB.Attr.Status)==9:
#             print(np.linalg.cond(H))
        x_est = []
        for nnn in solution:
             x_est.append(solution[nnn])
        results.append(x_est)
    return results, np.array(status)