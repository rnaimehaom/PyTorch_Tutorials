""" 
Name:           PyTorch Transfer Learning Tutorials
Script Name:    03_fine_tune_trained.py
Author:         Gary Hutson
Date:           05/09/2022
Usage:          python 02_train_fe.py

"""

import transferlearner.config as cfg
from transferlearner.utils.data import get_dataloader

from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import time
import os


# Augment the image using transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(cfg.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
])

valid_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
])

# Create our data loaders
(train_ds, train_dl) = get_dataloader(cfg.TRAIN,
	custom_transforms=train_transform,
	batch_size=cfg.FINETUNE_BATCH_SIZE)

(valid_ds, valid_dl) = get_dataloader(cfg.VAL,
	custom_transforms=valid_transform,
	batch_size=cfg.FINETUNE_BATCH_SIZE, random_shuffle=False)



# Load up our model backbone
model = resnet50(pretrained=True)
num_feats = model.fc.in_features

# Get the models and freeze one of the layers
for module, param in zip(model.modules(), model.parameters()):
    if isinstance(module, nn.BatchNorm2d):
        param.requires_grad = False
        
        
        
# Define what the head of the network should look like and attach it
