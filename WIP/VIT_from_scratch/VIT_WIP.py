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


# Create method for seperating each image into patches 
def patcher(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patch method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches




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
        
        self.ln1= nn.LayerNorm((self.n_patches **2 + 1, self.hidden_d))
        
        # Multi=headed Self Attention and classification token
        
        # Implement self.msa 
        
        # Layer normalisation 2
        self.ln2 = nn.LayerNorm((self.n_patches ** 2 + 1, self.hidden_d))
        
        # Encoder MLP
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU()
        )
        
        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1))
        
                
    def forward(self, images):
        # Dividing images into patches
        n,c,w,h = images.shape
        patches = patcher(images, self.n_patches)
        
        # Run linear for tokenization process
        tokens = self.linear_mapper(patches)
        
        # Add classification token to the token
        tokens = torch.stack([torch.vstack])
        










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
            
    