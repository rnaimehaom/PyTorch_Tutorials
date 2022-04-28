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
class LayerNormalise(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Create the MLP for the fully connected layer
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        return x

# Create the attention head
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 #1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim *3, bias=True) #Weight query, key and value for each vector
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b,n,_, h = *x.shape, self.heads
        qkv = self.to_qkv(x) # gets q = Q = Wq matmul x1, k=Wk nm x2, v= Wv mm x3
        q,k,v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1,0), value=True)
            assert mask.shape[-1] == dots.shape[-1], '[MASK DIM ERROR] the mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:,:, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out


# Create the transformer block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalise(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalise(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))
    
    def forward(self, x, mask=None):
        for attention, mlp in self.layers: 
            x = attention(x, mask=mask) #go to attention
            x = mlp(x)
        return x


# Create the image transformer

class ImageTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
            super().__init__()
            assert image_size % patch_size == 0, '[IMAGE DIM ERROR] image dimensions mist be divisible by the patch size of the image'
            num_patches = (image_size // patch_size) ** 2 # e.g. (32/4)**2= 64
            patch_dim = channels * patch_size ** 2 # e.g. 3*8**2 = 64*3
            self.patch_size = patch_size
            self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
            # Standard deviation of the gaussian initialised to .02 as per
            # an image is worth 16 * 16 words
            torch.nn.init.normal_(self.pos_embedding, std=.02)
            self.patch_conv = nn.Conv2d(3, dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
            #self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)
            self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
            self.dropout = nn.Dropout(emb_dropout)
            # Call our previous class, the transformer to pass to 
            # multiple attention heads and pos encodings
            # through patch extraction similar to how NLP text is processed
            # and masked
            self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
            self.to_cls_token = nn.Identity()




