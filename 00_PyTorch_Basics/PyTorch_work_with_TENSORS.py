import torch
import numpy as np

def printer(func):
    def func():
        pass
    pass


# Create a tensor
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(type(x_data))
print(data, '\n',x_data)

# Create a PyTorch tensor from numpy array
np_array = np.array(data)
x_numpy = torch.from_numpy(np_array)
print(x_numpy)

# From a different tensor
x_ones = torch.ones_like(x_data)
print(f'Ones tensor:\n{x_ones}\n')
# Create a tensor from a random function
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'Random Tensor: \n {x_rand} \n')

# Specify dimensions of a tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random tensor: \n {rand_tensor} \n')
print(f'Ones tensor: \n {ones_tensor} \n')
print(f'Zeros tensor: \n {zeros_tensor}')

# Tensor attributes
tensor = torch.rand(3,4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')


# Tensor operations

if torch.cuda.is_available():
   tensor = tensor.to('cuda')
   
# Indexing and slicing
tensor = torch.ones(4,4)
print(f'First row: {tensor[0]}')
print(f'First column: {tensor[:,0]}')
print(f'Last column: {tensor[...,-1]}') 
tensor[:,1]=0
print(tensor)

# Joining tensors together
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic on tensors
y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
print(torch.matmul(tensor, tensor.T, out=y3))

# Element wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(torch.mul(tensor, tensor, out=z3))

# Single element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In place operations
tensor.add_(5)
print(tensor)

# Casting operations
t = torch.ones(10)
nump = t.numpy()
print(nump)

# A change in the tensor affects the numpy array
t.add_(1)
print(t)

# Numpy array to tensor
n = np.ones(10)
t = torch.from_numpy(n)
np.add(n,1,out=n)
print(t, '\n', n)





