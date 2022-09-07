""" 
Name:           PyTorch Transfer Learning Tutorials
Script Name:    04_Model_Inference.py
Author:         Gary Hutson
Date:           07/09/2022
Usage:          python 04_Model_Inference.py

"""

import transferlearner.config as cfg
from transferlearner.utils.data import get_dataloader
from torchvision import transforms
from imutils import paths
from torch import nn
import matplotlib.pyplot as plt
import argparse

# Create the arguments to parse our arguments

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model_path', required=True, 
                help='the path to the serialised and trained model')
# Roll up the arguments to be used for the argument parser



