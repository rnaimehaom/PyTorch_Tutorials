# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# Use a test picture to work with 

img = Image.open('images/cat1.jpeg')
fig = plt.figure()
print(img)


# Transform the image to a PyTorch tensor
transform = Compose([Resize((224,224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0)
print(x.shape)


# Use einops to flatten the images into patches
patch_size = 16
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

# Create a patch embedding class to keep everything neet

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, emb_size: int=768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # Break the image down into s1 x s2 patches and then flatten the image
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))

    def forward(self, x:Tensor) -> Tensor:
        b,_,_,_ = x.shape
        x = self.projection(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        return x
    
    
print(PatchEmbedding()(x).shape)














