""" 
Name:       PyTorch VIT from scratch for Computer Vision Classification
Author:     Gary Hutson
Date:       10/08/2022
Purpose:    Python code to show how to train a VIT from scratch
"""

# Get our imports
import time
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

# Get our custom imports
from VisionTransformer.model_components import *

# Set our training params
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
N_EPOCHS = 150

# Preprocess function
DL_PATH = "C:\Pytorch\Spyder\CIFAR10_data"

# Preprocess dataset

transform = torchvision.transforms.Compose(
     [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)

# Train the model

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')



# Use our model

model = ImageTransformer(image_size=32, patch_size=4, num_classes=10, 
                         channels=3,dim=64, depth=6, att_heads=8, 
                         mlp_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)



# Save your trained model

PATH = ".\ViTnet_Cifar10_4x4_aug_1.pt" # Use your own path
torch.save(model.state_dict(), PATH)


# # =============================================================================
# # model = ViT()
# # model.load_state_dict(torch.load(PATH))
# # model.eval()            
# # =============================================================================
