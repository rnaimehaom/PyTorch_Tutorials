import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read in the medical insurance data
df = pd.read_csv('https://raw.githubusercontent.com/StatsGary/Data/main/insurance.csv')
print(df.head())
print(df['charges'].describe())

# Encode the categorical features
cat_cols = ['sex', 'smoker', 'region', 'children']

