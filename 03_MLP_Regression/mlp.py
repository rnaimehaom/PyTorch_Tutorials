import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

batch_size = 60000

#=====================================================================================
# Data Loading
#=====================================================================================

# Read in the medical insurance data
df = pd.read_csv('https://raw.githubusercontent.com/StatsGary/Data/main/insurance.csv')

# Drop nulls
df.dropna(axis='columns',inplace=True)

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
print(cats[:5])

# Convert continuous variables to a tensor
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y].values, dtype=torch.float).reshape(-1,1)

# Set embedding sizes
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

class MLPRegressor(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


# Use the model
torch.manual_seed(123)
model = MLPRegressor(emb_szs, conts.shape[1], out_sz=1, layers=[200,100], p=0.4)
print(model)

#=====================================================================================
# Split the data
#=====================================================================================
test_size = int(batch_size * .2)
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

print(cat_train.shape)

#=====================================================================================
# Train the model
#=====================================================================================

def train(y_train, categorical_train, continuous_train, learning_rate=0.001, epochs=300,
          print_out_interval=2):

    criterion = nn.MSELoss()  # we'll convert this to RMSE later
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()

    losses = []

    for i in range(epochs):
        i+=1 #Zero indexing trick to start the print out at epoch 1
        y_pred = model(categorical_train, continuous_train)
        loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
        losses.append(loss)
        
        # a neat trick to save screen space:
        if i%print_out_interval == 1:
            print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    return losses, model

losses, model = train(y_train, cat_train, con_train, learning_rate=0.001, epochs=5000, print_out_interval=100)