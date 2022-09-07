""" 
Name:           Optimisation with linear models - logistic reg
Track:          PyTorch Transfer Learning Tutorials
Script Name:    1_optim_with_linear_models.py
Author:         Gary Hutson
Date:           05/09/2022
Usage:          python 1_optim_with_linear_models.py
"""

# Define imports
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize
# Bring in custom imports
from AUROCker.functions import sigmoid, AUROC, convex_AUROC, index_I0I1, derivative_chained_AUROC

# Constants
TOP_PROP = 0.9
SPLIT_PROP = 0.2

# Load in test dataset
X, y = load_boston(return_X_y=True)
# Binarise the y outcome variable
y = np.where(y > np.quantile(y,TOP_PROP), 1,0)

# Simulate a number of datasets
num_to_sim = 200
auc_list = []
weight_list = []
winit = np.repeat(0, X.shape[1])
print(winit)

# Initialise a loop to loop through our simulations

for sim in range(num_to_sim):
    y_train, y_test, X_train, X_test = train_test_split(y, X,
                                                        test_size=SPLIT_PROP,
                                                        random_state=sim,
                                                        stratify=y)
    # Scale the dataset 
    scaled = StandardScaler().fit(X_train)
    idx0_train, idx1_train = index_I0I1(y_train)
    idx0_test, idx1_test = index_I0I1(y_test)
    
    # Optimised AUC
    optim_AUC = minimize(func=convex_AUROC, x0=winit,
                         args=(scaled.transform(X_train),
                               idx0_train, idx1_train),
                         method='L-BFGS-B', 
                         jac=derivative_chained_AUROC).x
    
    eta_auc = scaled.transform(X_test).dot(optim_AUC)
    logit_mod = LogisticRegression(penalty='none')
    probs_logit = logit_mod.fit(scaled.transform(X_train), y_train).predict_proba(X_test)[:,1]
    
    
    



