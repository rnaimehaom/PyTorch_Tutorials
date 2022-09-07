""" 
Name:           PyTorch Transfer Learning Tutorials
Script Name:    04_Model_Inference.py
Author:         Gary Hutson
Date:           07/09/2022
Usage:          
                python 04_Model_Inference.py --model_path model_output/flower_finetune_model.pth
                python 04_Model_Inference.py --model_path model_output/flower_warmup_model.pth
"""

import transferlearner.config as cfg
from transferlearner.utils.data import get_dataloader
from torchvision import transforms
from imutils import paths
from torch import nn
import torch
import matplotlib.pyplot as plt
import argparse

# Create the arguments to parse our arguments

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model_path', required=True, 
                help='the path to the serialised and trained model')
# Roll up the arguments to be used for the argument parser
args = vars(ap.parse_args())

# Create the transforms
infer_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
])

# Calculate the inverse of the transformations applied to RGB colour channels

inverse_mean = [-m/s for (m,s) in zip(cfg.MEAN, cfg.STD)]
inverse_std = [1/s for s in cfg.STD]

# Define the inverse transform to convert the RGB
inverse_trans_normalise = transforms.Normalize(mean=inverse_mean, std=inverse_std)

# Grab the test data loaders
(test_ds, test_dl) = get_dataloader(cfg.VAL,
	custom_transforms=infer_transform,
	batch_size=cfg.PRED_BATCH_SIZE)


# Grab CUDA and activate if available
if torch.cuda.is_available():
    loc_mapping = lambda storage, loc: storage.cuda
else:
    loc_mapping = 'cpu'
    

# Load the model from the argparse path specified
print('(MODEL LOADING) loading the serialised and pretrained model')
model = torch.load(args['model_path'], map_location=loc_mapping)

# Move the model to our device and set the model into evaluation mode
model.to(cfg.DEVICE)
model.eval()

# Grab a batch of data using a generator
batch = next(iter(test_dl))
(images, labels) = (batch[0], batch[1])

# Initialise a chart to use
fig = plt.figure('Results', figsize=(10,10))

# Switch off the automatic updating of gradients, as this is an inference script
with torch.no_grad():
    # Send the images onwards to the device
    images = images.to(cfg.DEVICE)
    # Make the preds
    print('(INFER) performing inference...')
    preds = model(images)
    # Loop over all the images in the batch
    for i in range(0, cfg.PRED_BATCH_SIZE):
        ax = plt.subplot(cfg.PRED_BATCH_SIZE, 1, i + 1)
        image = images[i]
        image = inverse_trans_normalise(image).cpu().numpy()
        image = (image * 255).astype('uint8')
        image = image.transpose((1,2,0))
        
        # Get the ground truth actual class label
        idx = labels[i].cpu().numpy()
        gt_label = test_ds.classes[idx]
        
        # Get the predicted label 
        pred = preds[i].argmax().cpu().numpy()
        pred_label = test_ds.classes[pred]
        
        # Add the results and the image to the plot
        graph_info = 'Predicted label: {} vs Ground Truth: {}'.format(
            pred_label, gt_label
        )
        plt.imshow(image)
        plt.title(graph_info)
        plt.axis('off')
        
    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.savefig('model_output/inference_plot.png')
        