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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# Bring in custom imports
from AUROCker.functions import sigmoid, AUROC, convex_AUROC, index_I0I1, derivative_chained_AUROC

# Constants
TOP_PROP = 0.9
SPLIT_PROP = 0.2

# Load in test dataset
