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

cat_sizes = [len(df[col].cat.categories) for col in cat_cols]
print(cat_sizes)
# Get embedding sizes
emb_szs = [(size, min(50,(size+1)//2)) for size in cat_sizes]
print(emb_szs)

# CONVERT CONTINUOUS VARIABLES
cont_cols = np.stack([df[col].values for col in cont_cols], axis=1)
cont_cols = torch.tensor(cont_cols, dtype=torch.float)
print(cont_cols)

#CONVERT TARGET (Y) LABEL
y= torch.tensor(df[y].values,dtype=float)
print(cats.shape, cont_cols.shape, y.shape)

#=====================================================================================
# 
#=====================================================================================
selfembeds = nn.ModuleList([nn.Embedding(ne,nf) for ne, nf in emb_szs])
print(selfembeds)

# Create the torch model
class MLPRegressor(nn.Module):

    def __init__(self, embed_size, n_continuous, layers, 
                output_size=1, drop_out_prob=0.5):
        # Sub class the nn.Module
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embed_size])
        self.dropout = nn.Dropout(drop_out_prob)
        self.norm_cont = nn.BatchNorm1d(n_continuous)

        # Layer list
        layer_list = []
        numb_embeds = [nf for ni, nf in emb_szs]
        numb_inputs = numb_embeds + n_continuous

        for i in layers:
            layer_list.append(nn.Linear(numb_inputs, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(drop_out_prob))
            numb_inputs = i

        layer_list.append(nn.Linear(layers[-1], output_size)) #Default output to 1 as regression problem
        self.layers = nn.Sequential(*layer_list)


    def forward(self, x_cats, x_conts):
        embeds_list = []

        for idx, emb in enumerate(self.embeds):
            embeds_list.append(emb(x_cats[:,idx]))

        x_variables = torch.cat(embeds_list,1)
        x_variables = self.dropout(x_variables)

        
