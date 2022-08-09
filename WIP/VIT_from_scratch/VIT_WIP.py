import numpy as np
import torch 
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

#https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Get example datasets from torchvision
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

np.random.seed(123)
torch.manual_seed(123)

# Setup the function 

def main():
    # Load data
    transform = ToTensor()
    
    train_set = MNIST('./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)
    
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=16)
    
    # Define the model
    
    model = ...
    N_EPOCHS=5
    LR=0.01
    
    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        # Iterate through batches
        
    