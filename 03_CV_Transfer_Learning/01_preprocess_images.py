""" 
Name:           PyTorch Transfer Learning Tutorials
Script Name:    01_preprocess_images.py
Author:         Gary Hutson
Date:           12/08/2022
Usage:          python 01_preprocess_images.py∂ß

"""

from transferlearner.utils.data import copy_images
import transferlearner.config as cfg
from imutils import paths
import numpy as np

image_pths = list(paths.list_images(cfg.DATA_PATH))
np.random.shuffle(image_pths)

# generate training and validation paths
valid_len = int(len(image_pths) * cfg.VAL_SPLIT)
train_len = len(image_pths) - valid_len
train_path = image_pths[:train_len]
valid_path = image_pths[train_len:]

# copy the training and validation images to their respective
# directories
print("[INFO] copying training and validation images...")
copy_images(train_path, cfg.TRAIN)
copy_images(valid_path, cfg.VAL)