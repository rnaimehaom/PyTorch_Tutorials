""" 
Name:           AUROC maximisation with PyTorch
Track:          PyTorch Transfer Learning Tutorials
Script Name:    2_AUROC_maxmisation_with_PyTorch.py
Author:         Gary Hutson
Date:           05/09/2022
Usage:          python 2_AUROC_maxmisation_with_PyTorch.py
Credits:        Code adapted from Erik Drysdale
"""

# Define imports
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# Bring in custom imports
from AUROCker.functions import sigmoid, AUROC, convex_AUROC, index_I0I1, derivative_chained_AUROC

# Constants
TOP_PROP = 0.9
SPLIT_PROP = 0.2

# Get the data
data = fetch_california_housing(download_if_missing=True)
cn_cali = data.feature_names
X_cali = data.data
y_cali = data.target
y_cali += np.random.randn(y_cali.shape[0])*(y_cali.std())
y_cali = np.where(y_cali > np.quantile(y_cali,0.95),1,0)
y_cali_train, y_cali_test, X_cali_train, X_cali_test = \
  train_test_split(y_cali, X_cali, test_size=0.2, random_state=1234, stratify=y_cali)
enc = StandardScaler().fit(X_cali_train)


# Create the training class

class feedy(nn.Module):
    def __init__(self,num_features):
      super(feedy, self).__init__()
      p = num_features
      self.fc1 = nn.Linear(p, 36)
      self.fc2 = nn.Linear(36, 12)
      self.fc3 = nn.Linear(12, 6)
      self.fc4 = nn.Linear(6,1)
    
    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return(x)

# Binary loss function
criterion = nn.BCEWithLogitsLoss()
# Seed the network
torch.manual_seed(1234)
nnet = feedy(num_features=X_cali.shape[1])
optimizer = torch.optim.Adam(params=nnet.parameters(),lr=0.001)

np.random.seed(1234)

y_cali_R, y_cali_V, X_cali_R, X_cali_V = \
  train_test_split(y_cali_train, X_cali_train, test_size=0.2, random_state=1234, stratify=y_cali_train)
enc = StandardScaler().fit(X_cali_R)

idx0_R, idx1_R = index_I0I1(y_cali_R)

nepochs = 100

auc_holder = []
for kk in range(nepochs):
  print('Epoch %i of %i' % (kk+1, nepochs))
  # Sample class 0 pairs
  idx0_kk = np.random.choice(idx0_R,len(idx1_R),replace=False) 
  for i,j in zip(idx1_R, idx0_kk):
    optimizer.zero_grad() 
    dlogit = nnet(torch.Tensor(enc.transform(X_cali_R[[i]]))) - \
        nnet(torch.Tensor(enc.transform(X_cali_R[[j]]))) 
    loss = criterion(dlogit.flatten(), torch.Tensor([1]))
    loss.backward() # backprop
    optimizer.step() # gradient-step
  # Calculate AUC on held-out validation
  auc_k = roc_auc_score(y_cali_V,
    nnet(torch.Tensor(enc.transform(X_cali_V))).detach().flatten().numpy())
  
  print(f'Current AUROC: {auc_k*100:.3f}%')
  if auc_k > 0.95:
    print('AUC > 90% achieved')
    break