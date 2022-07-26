from http.client import CannotSendHeader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#Custom model imports
from models.Regression import MLPRegressor

data_name ='medical_insurance'
# Read in the medical insurance data
df = pd.read_csv('https://raw.githubusercontent.com/StatsGary/Data/main/insurance_prod.csv')
# Drop nulls
df.dropna(axis='columns',inplace=True)
# Get number of rows
obs = len(df)

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

cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)
cats = torch.tensor(cats, dtype=torch.int64)
# Convert continuous variables to a tensor
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)
# Create outcome
y = torch.tensor(df[y].values, dtype=torch.float).reshape(-1,1)
# Set embedding sizes
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
#emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs = [(2, 1), (2, 1), (4, 2), (6, 3)]


# Instantiate inference model
model_infer = MLPRegressor(emb_szs, conts.shape[1], 1, [200,100], p=0.4)
model_infer.load_state_dict(torch.load('model_artifacts/medical_insurance_400.pt'))
print(model_infer.eval())


def prod_data(model, cat_prod, cont_prod, verbose=False):
    # Pass the inputs from the cont and cat tensors to function
    with torch.no_grad():
        y_val = model(cat_prod, cont_prod)
    
    # Get preds on prod data
    preds = [] 
    for i in range(len(cat_prod)):
        result = y_val[i].item()
        preds.append(result)
        
        if verbose == True:
            print(f'The predicted value is: {y_val[i].item()}')
        
    return preds
    
    
# Use prod data function
prod = prod_data(model_infer, cats, conts, verbose=True)

        
        

    