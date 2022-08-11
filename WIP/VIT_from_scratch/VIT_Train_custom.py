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
from torchvision import transforms 
from imutils import paths
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get our custom imports
from VisionTransformer.model_components import *
from VisionTransformer.data import get_and_copy_images, create_train_and_val_dirs, save_class_labels_to_json


# Set our training params
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
N_EPOCHS = 50
DATASET_PATH = 'images/flower_photos'
TRAIN_DIR_NAME = 'train'
VALID_DIR_NAME = 'valid'
VAL_SPLIT = 0.1
MODEL_NAME = 'flowers_VIT'
INPUT_WIDTH = 64
INPUT_HEIGHT = 64
PATCH_SIZE = 6
NUM_PATCHES = (INPUT_HEIGHT // PATCH_SIZE) ** 2
SAVE_PATH = "models" 
MODEL_NAME = 'flower_VIT'
LR = 0.001

# Take the original files and create splits
create_train_and_val_dirs(DATASET_PATH, 0.1, 
                          train_dir_name=f'images/{TRAIN_DIR_NAME}',
                          val_dir_name=f'images/{VALID_DIR_NAME}')


# Create our data transforms
transforms_train = transforms.Compose(
     [
        transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH)),
        # Horizontal flip
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()]
     )

transforms_valid = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH)),
        torchvision.transforms.ToTensor()
    ]
)


# Read our datasets from the storage folder 
train_dataset = ImageFolder(root=f'images/{TRAIN_DIR_NAME}', transform=transforms_train)
valid_dataset = ImageFolder(root=f'images/{VALID_DIR_NAME}', transform=transforms_valid)

print(type(train_dataset))

print(f'[TRAIN INFO] the training set contains n= {len(train_dataset)} of examples')
print(f'[VALID INFO] the validation set contains n= {len(valid_dataset)} of examples')

# Create the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_TEST)

# Print the classes in the dataset
save_class_labels_to_json(train_dataset)

# Create functions to train and evaluate model
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
            
# Evaluation function
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



# # Use our model
model = ImageTransformer(image_size=INPUT_HEIGHT, patch_size=4, num_classes=5, 
                         channels=3,dim=64, depth=6, att_heads=8, 
                         mlp_dim=128, large_network=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_dataloader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, valid_dataloader, test_loss_history)


# Save your trained model
FULL_PATH = f'{SAVE_PATH}/{MODEL_NAME}.pt'
torch.save(model.state_dict(), FULL_PATH)


# # # =============================================================================
# # # model = ViT()
# # # model.load_state_dict(torch.load(PATH))
# # # model.eval()            
# # # =============================================================================
