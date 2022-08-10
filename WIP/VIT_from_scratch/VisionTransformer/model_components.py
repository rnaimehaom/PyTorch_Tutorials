import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torchvision.transforms as T

class Residualiser(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.normalisation = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.normalisation(x), **kwargs)

class MultiLayerPercep(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        # Define the activation function as Gaussian Error Linear Units
        self.af1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.drop1(x)
        x = self.nn2(x)
        x = self.drop2(x)
        
        return x

class AttentionHead(nn.Module):
    def __init__(self, dim, att_heads = 8, dropout = 0.1):
        super().__init__()
        self.att_heads = att_heads
        self.scale = dim ** -0.5 

        ## Wq,Wk,Wv for each vector
        self.to_query_key_value = nn.Linear(dim, dim * 3, bias = True) 
        torch.nn.init.xavier_uniform_(self.to_query_key_value.weight)
        # Initialise an empty tensor with zeros
        torch.nn.init.zeros_(self.to_query_key_value.bias)
        
        # Map to a linear layer
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.drop1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.att_heads
        # The below gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qkv = self.to_query_key_value(x) 
        # Get our multi head attentions
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) 
        # Short hand einstein equation processing
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has the incorrect dimension size'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) # Take the dot product of the softmax
        # Get the product of v times whatever resides in the softmax
        out = torch.einsum('bhij,bhjd->bhid', attn, v) 
        # Concatentate the attention heads into one matrix, ready for next encoder block
        out = rearrange(out, 'b h n d -> b n (h d)') 
        out =  self.nn1(out)
        out = self.drop1(out)
        return out

# Implement our transformer object
class Transformer(nn.Module):
    def __init__(self, dim, depth, att_heads, mlp_dim, dropout):
        super().__init__()
        # Create an empty module list as the layers of the transformer
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residualiser(LayerNormalize(dim, AttentionHead(dim, att_heads = att_heads, dropout = dropout))),
                Residualiser(LayerNormalize(dim, MultiLayerPercep(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, multi_layer_percep in self.layers:
            # Go to the attention head
            x = attention(x, mask = mask) 
            # Get the Multi Layer Perceptron
            x = multi_layer_percep(x) 
        return x

class ImageTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, 
                 depth, att_heads, mlp_dim, channels = 3, 
                 dropout = 0.1, emb_dropout = 0.1, large_network=False):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  
        patch_dim = channels * patch_size ** 2 
        
        self.large_network = large_network
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        # The below is initialised with a standard deviation aligning to the transformers research
        torch.nn.init.normal_(self.pos_embedding, std = .02) 
        self.patch_conv= nn.Conv2d(3,dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
        # Define sentence-level classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) 
        self.dropout = nn.Dropout(emb_dropout)
        # Call our custom Transformer class and pass through the relevant parameters
        self.transformer = Transformer(dim, depth, att_heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        # For finetuning the model just use a linear layer
        self.nn1 = nn.Linear(dim, num_classes)  
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
        if self.large_network==True:
            # Add our additional hidden layers for really large datasets
            self.af1 = nn.GELU() 
            self.drop1 = nn.Dropout(dropout)
            # Add the second layer
            self.nn2 = nn.Linear(mlp_dim, num_classes)
            torch.nn.init.xavier_uniform_(self.nn2.weight)
            torch.nn.init.normal_(self.nn2.bias)
            self.drop2 = nn.Dropout(dropout)

    def forward(self, img, mask = None):
        p = self.patch_size
        x = self.patch_conv(img) 
        x = rearrange(x, 'b c h w -> b (h w) c') 

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        
        if self.large_network == True:
            x = self.af1(x)
            x = self.do1(x)
            x = self.nn2(x)
            x = self.drop2(x)
        
        return x
