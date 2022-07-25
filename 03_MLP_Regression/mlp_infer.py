import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#Custom model imports
from models.Regression import MLPRegressor

#Use embedding sizes from training step
emb_szs = [(2, 1), (2, 1), (4, 2), (6, 3)]

# Instantiate inference model
model_infer = MLPRegressor(emb_szs, 2, 1, [200,100], p=0.4)
model_infer.load_state_dict(torch.load('model_artifacts/medical_insurance_400.pt'))
print(model_infer.eval())