from os import device_encoding
import numpy as np
import torch 
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

#https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py
# Get example datasets from torchvision
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# For repeatability
np.random.seed(123)
torch.manual_seed(123)

# Create VIT class

class VisionTransformerVIT(nn.Module):
    def __init__(self, input_shape, n_patches=7, hidden_d=8, n_heads=2, out_d=10, device=None): 
        super(VisionTransformerVIT, self).__init__()
        self.device = device
        
        # Input shape and patches
        self.input_shape = input_shape
        self.n_patches = n_patches
        self.n_heads = n_heads
        
        assert input_shape[1] % n_patches == 0, 'Input shape not entirely divisible by the number of patches in the image'
        assert input_shape[2] % n_patches ==0, 'Input shape not entirely divisible by the number of patches in the image'
        
        self.patch_size = (input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.hidden_d = hidden_d
        
        # Linear mapping
        self.input_d = int(input_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Position embedding 
        # Refer to forward method
        
        self.layer_norm1 = nn.LayerNorm((self.n_patches **2 + 1, self.hidden_d))
        
        # Multi=headed Self Attention and classification token
        
        # Implement self.msa 
        
        # Layer normalisation 2
        self.ln2 = nn.LayerNorm((self.n_patches **2 + 1, self.hidden_d))
        
        # Encoder MLP
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d)
        )
        
        
        
        
                
    def forward(self, images):
        pass










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
        for batch in train_dataloader:
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat,y) / len(x)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}')
            
    