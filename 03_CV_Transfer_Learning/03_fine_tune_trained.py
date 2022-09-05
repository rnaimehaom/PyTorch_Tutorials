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

validation_transform = transforms.Compose([
    
])