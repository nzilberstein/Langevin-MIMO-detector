import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
from gurobipy import *
from multiprocessing.dummy import Pool as ThreadPool 

import yaml
import os
from numpy import linalg as LA

import sys
sys.path.append(os.path.dirname(os.getcwd()))


#\\Own libraries
from utils.util import *
from pathlib import Path

dirPath = os.path.dirname(os.getcwd())
with open(dirPath + '/config.yml', 'r') as f:
    aux = yaml.load(f,  Loader=yaml.FullLoader)
config = dict2namespace(aux)

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
    return results



def ml_proc(hBatch, yBatch):
    
    sigConst = sigConst = np.linspace(-config.M+1, config.M-1, config.M) 
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts
    shatBatch = mlSolver(np.array([hBatch]), np.array([yBatch]), sigConst)
    return shatBatch

def ml_proc_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return ml_proc(*a_b)
