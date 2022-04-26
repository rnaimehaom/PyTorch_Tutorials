import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# Create the residual skip connection layer
class Residual(nn.Module):
    # Create initialisation function
    def __init__(self, fn):
        super().__init__() #Inherit from the nn Module to get all the lovely parameters
        self.fn = fn
    def forward(self,x, **kwargs):
        return self.fn(x, **kwargs) + x

# Create normalisation layer class
class LayerNormalise(nn.Module)
