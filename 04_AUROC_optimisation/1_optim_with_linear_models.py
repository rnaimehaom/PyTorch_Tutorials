""" 
Name:           Optimisation with linear models - logistic reg
Track:          PyTorch Transfer Learning Tutorials
Script Name:    1_optim_with_linear_models.py
Author:         Gary Hutson
Date:           05/09/2022
Usage:          python 1_optim_with_linear_models.py
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

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
# Binarise the y outcome variable
y = np.where(y > np.quantile(y,TOP_PROP), 1,0)

# Simulate a number of datasets
num_to_sim = 100
auc_list = []
results_list = []
winit = np.repeat(0, X.shape[1])

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
    optim_AUC = minimize(fun=convex_AUROC, x0=winit,
                         args=(scaled.transform(X_train),
                               idx0_train, idx1_train),
                         method='L-BFGS-B', 
                         jac=derivative_chained_AUROC).x
    
    eta_auc = scaled.transform(X_test).dot(optim_AUC)
    logit_mod = LogisticRegression(penalty='none')
    probs_logit = logit_mod.fit(scaled.transform(X_train), y_train).predict_proba(X_test)[:,1]
    auc1, auc2 = roc_auc_score(y_test, eta_auc), roc_auc_score(y_test, probs_logit)
    auc_list.append([auc1, auc2])
    # results_list.append(pd.DataFrame(
    #     {'cn':load_boston()['feature_names'],
    #      'auc':optim_AUC,'logit':logit_mod.coef_.flatten()}
    # ))
    
auc_diff = np.vstack(auc_list).mean(axis=0)
print('AUC from convex AUROC: %0.2f%%\nAUC for LogisticRegression: %0.2f%%' % 
      (auc_diff[0], auc_diff[1]))


# Visualise the difference
scaler = StandardScaler().fit(X)
idx0, idx1 = index_I0I1(y)
optim_cAUC = minimize(fun=convex_AUROC, x0=winit,
                      args=(scaler.transform(X), idx0, idx1),
                      method='L-BFGS-B', jac=derivative_chained_AUROC).x
eta_auc = scaler.transform(scaler.transform(X)).dot(optim_cAUC)
logistic_mod = LogisticRegression(max_iter=1e3).fit(scaler.transform(scaler.transform(X)),
                                                    y).predict_log_proba(scaler.transform(X))[:,1]

output_dataframe = pd.DataFrame({
    'y':y, 
    'Logistic': logistic_mod, 
    'cAUROC': eta_auc
}).melt('y')

print(output_dataframe)

# Create facetgrid to compare the AUROC outputs
graph = sns.FacetGrid(data=output_dataframe, hue='y', col='variable', sharex=False, sharey=False, height=5)
graph.map(sns.distplot, 'value')
graph.add_legend()
plt.savefig('output.png')