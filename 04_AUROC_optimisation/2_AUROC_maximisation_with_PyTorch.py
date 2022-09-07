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
# Bring in custom imports
from AUROCker.functions import sigmoid, AUROC, convex_AUROC, index_I0I1, derivative_chained_AUROC

# Constants
TOP_PROP = 0.9
SPLIT_PROP = 0.2

# Load in test dataset
data = fetch_california_housing(download_if_missing=True)
feat_names_cali = data.feature_names
print(feat_names_cali)

# Split data 
X = data.data
y = data.target
y += np.random.randn(y.shape[0])*(y.std())
y = np.where(y > np.quantile(y, 0.95),1,0)

# Create the splits
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=SPLIT_PROP, stratify=y)
scaled = StandardScaler().fit(X_train)


# Create the feedforward network 
class feedy(nn.Module):
    def __init__(self, num_features):
        super(feedy, self).__init__()
        feats = num_features
        self.fc1 = nn.Linear(feats, 36)
        self.fc2 = nn.Linear(36,12)
        self.fc3 = nn.Linear(12,6)
        self.fc4 = nn.Linear(6,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return X
    
# Binary loss function
criterion = nn.BCEWithLogitsLoss()
nnet = feedy(num_features=X.shape[1])
optimizer = torch.optim.Adam(params=nnet.parameters(), lr=0.001)

# Validation and training samples
y_held, y_valid, x_held, x_valid = train_test_split(y_train, X_train, test_size=SPLIT_PROP, stratify=y_train)
scaler = StandardScaler().fit(x_held)
idx0_held, idx1_held = index_I0I1(y_held)

# Set the epochs
epochs = 100
auc_history = []

for epoch in range(epochs):
    print('[EPOCH] {} of {}'.format(epoch + 1, epochs))
    # Sample class 0 pairs
    idx0_epoch = np.random.choice(idx0_held)
    