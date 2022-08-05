
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from datetime import datetime as dt
# Custom imports
from models.Regression import MLPRegressor

#=====================================================================================
# Data Loading
#=====================================================================================
data_name ='medical_insurance'

# Read in the medical insurance data
df = pd.read_csv('https://raw.githubusercontent.com/StatsGary/Data/main/insurance.csv')

# Drop nulls
df.dropna(axis='columns',inplace=True)

# Get number of rows
obs = len(df)

# Divide obs in half to get half batch size
batch_size = obs // 2

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

y = torch.tensor(df[y].values, dtype=torch.float).reshape(-1,1)

# Set embedding sizes
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
print(conts.shape[1])

# Use the model
torch.manual_seed(123)
model = MLPRegressor(emb_szs, conts.shape[1], out_sz=1, layers=[200,100], p=0.4)
print('[INFO] Model definition')
print(model)
print('='* 80)

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

#=====================================================================================
# Train the model
#=====================================================================================

def train(model, y_train, categorical_train, continuous_train,
          y_val, categorical_valid, continuous_valid,
          learning_rate=0.001, epochs=300, print_out_interval=2):

    global criterion
    criterion = nn.MSELoss()  # we'll convert this to RMSE later
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    model.train()

    losses = []
    preds = []

    for i in range(epochs):
        i+=1 #Zero indexing trick to start the print out at epoch 1
        y_pred = model(categorical_train, continuous_train)
        preds.append(y_pred)
        loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
        losses.append(loss)
        
        if i%print_out_interval == 1:
            print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('='*80)
    print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
    print(f'Duration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    # Evaluate model
    with torch.no_grad():
        y_val = model(categorical_valid, continuous_valid)
        loss = torch.sqrt(criterion(y_val, y_test))
    print(f'RMSE: {loss:.8f}')

    # Create empty list to store my results
    preds = []
    diffs = []
    actuals = []

    for i in range(len(categorical_valid)):
        diff = np.abs(y_val[i].item() - y_test[i].item())
        pred = y_val[i].item()
        actual = y_test[i].item()

        diffs.append(diff)
        preds.append(pred)
        actuals.append(actual)

    valid_results_dict = {
        'predictions': preds,
        'diffs': diffs,
        'actuals': actuals
    }

    # Save model
    torch.save(model.state_dict(), f'model_artifacts/{data_name}_{epochs}.pt')
    # Return components to use later
    return losses, preds, diffs, actuals, model, valid_results_dict, epochs


# Use the training function to train the model

losses, preds, diffs, actuals, model, valid_results_dict, epochs = train(
            model=model, y_train=y_train, 
            categorical_train=cat_train, 
            continuous_train=con_train,
            y_val=y_test, 
            categorical_valid=cat_test,
            continuous_valid=con_test,
            learning_rate=0.01, 
            epochs=400, 
            print_out_interval=25)

#=====================================================================================
# Validate the model
#=====================================================================================
valid_res = pd.DataFrame(valid_results_dict)

# Visualise results
current_time = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.figure()
sns.scatterplot(data=valid_res, 
                x='predictions', y='actuals', size='diffs', hue='diffs')#, palette='deep')
plt.savefig(f'charts/{data_name}valid_results_{current_time}.png')

# Produce validation graph
    
losses_collapsed = [losses[i].item() for i in range(epochs)]
epochs = [ep+1 for ep in range(epochs)]
eval_df = pd.DataFrame({
    'epochs': epochs,
    'loss': losses_collapsed
})

# Save data to csv
eval_df.to_csv(f'data/{data_name}_valid_data_{current_time}.csv', index=None)

# Create SNS chart
plt.figure()
palette = sns.color_palette("mako_r", 6)
sns.lineplot(data=eval_df, x='epochs', y='loss', palette=palette)
plt.savefig(f'charts/{data_name}_loss_chart_{current_time}.png')