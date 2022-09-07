import numpy as np
from scipy.optimize import minimize

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def index_I0I1(y):
    return ((np.where(y==0)[0],
             np.where(y==1)[0]))
    
def AUROC(eta, idx0, idx1):
    denominator = len(idx0) * len(idx1)
    # Initialise an empty numeric
    num = 0
    for i in idx1:
        num += sum(eta[i] > eta[idx0]) + 0.5*sum(eta[i]==eta[idx0])
        
    return (num/denominator)


def convex_AUROC(w, X, idx0, idx1):
    eta = X.dot(w)
    denominator = len(idx0) * len(idx1)
    num = 0
    for i in idx1:
        num += sum(np.log(sigmoid(eta[i] - eta[idx0])))
        
    return (-num / denominator)

def derivative_chained_AUROC(w, X, idx0, idx1):
    eta = X.dot(w)
    n0, n1 =  len(idx0), len(idx1)
    den = n0 * n1
    num = 0
    for i in idx1:
        num += ((1 - sigmoid(eta[i] - eta[idx0])).reshape([n0,1]) * (X[[i]] - X[idx0]) ).sum(axis=0) # *
    return( - num / den)
    