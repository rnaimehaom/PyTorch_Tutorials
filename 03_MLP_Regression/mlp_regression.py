import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#=====================================================================================
# Data Loading
#=====================================================================================

# Read in the medical insurance data
df = pd.read_csv('https://raw.githubusercontent.com/StatsGary/Data/main/insurance.csv')

#=====================================================================================
# Feature Engineering
#=====================================================================================
# Encode the categorical features
cat_cols = ['sex', 'smoker', 'region', 'children']
cont_cols = ['age', 'bmi']

# Set the target (y) column
y = ['charges']

# CONVERT CATEGORICAL COLUMNS
for cat in cat_cols:
    df[cat] = df[cat].astype('category')

# Get the cat codes for each of the categorical values
cats = np.stack([df[col].cat.codes.values for col in cat_cols],axis=1)

# Convert this to a tensor
cats = torch.tensor(cats, dtype=torch.int64)
print(cats)

# CONVERT CONTINUOUS VARIABLES
cont_cols = np.stack([df[col].values for col in cont_cols], axis=1)
cont_cols = torch.tensor(cont_cols, dtype=torch.float)
print(cont_cols)

#CONVERT TARGET (Y) LABEL
y= torch.tensor(df[y].values,dtype=float)
print(cats.shape, cont_cols.shape, y.shape)
