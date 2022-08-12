""" 
Name:           PyTorch Transfer Learning Tutorials
Script Name:    02_train_fe.py
Author:         Gary Hutson
Date:           12/08/2022
Usage:          python 02_train_fe.py

"""

# import the necessary packages
import transferlearner.config as cfg
from transferlearner.utils.data import get_dataloader
from transferlearner.graphs import visualise

# Import the model backbone to use for transfer learning
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

#-------------------------------------------------------------------------
# Data augmentation and loader steps
#-------------------------------------------------------------------------

train_transform = transforms.Compose([
	transforms.RandomResizedCrop(cfg.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	# Normalise based on the mean and SD of the RGB pixel distribution
	transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
])
valid_transform = transforms.Compose([
	transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
])

# Dataloader creation
(train_ds, train_dl) = get_dataloader(cfg.TRAIN,
	custom_transforms=train_transform,
	batch_size=cfg.FEATURE_EXTRACTION_BATCH_SIZE)

(valid_ds, valid_dl) = get_dataloader(cfg.VAL,
	custom_transforms=valid_transform,
	batch_size=cfg.FEATURE_EXTRACTION_BATCH_SIZE, random_shuffle=False)

#-------------------------------------------------------------------------
# Load RESNET50 model backbone
#-------------------------------------------------------------------------
model = resnet50(pretrained=True)

# Feature extraction = requires_grad = False
# Meaning we are using this to extract features and use in another
# modelling context
for param in model.parameters():
	param.requires_grad = False


# Add our head to top of network
mod_out_feats = model.fc.in_features
model.fc = nn.Linear(mod_out_feats, len(train_ds.classes))
model = model.to(cfg.DEVICE)

#-------------------------------------------------------------------------
# Create training function
#-------------------------------------------------------------------------

def train(model, train_ds, valid_ds, train_dl, valid_dl, lr=0.001, batch_size=64, epochs=10, device='cuda'):
	loss_function = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
	# Steps per epoch calc
	training_steps = len(train_ds) // batch_size
	valid_steps = len(valid_ds) // batch_size
	
	# Create a history similar to tensorflow
	history = {"train_loss": [], "train_acc": [], "val_loss": [],
		"val_acc": []}

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	for e in tqdm(range(epochs)):
     
		#--------------------- Train mode -------------------------------
		model.train()
		tot_train_loss = 0
		tot_valid_loss = 0

		# initialize the number of correct predictions in the training
		# and validation step
		train_correct = 0
		val_correct = 0

		for (i, (x, y)) in enumerate(train_dl):
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))

			# perform a forward pass and calculate the training loss
			pred = model(x)
			loss = loss_function(pred, y)
   
			# calculate the gradients
			loss.backward()

			# check if we are updating the model parameters and if so
			# update them, and zero out the previously accumulated gradients
			if (i + 2) % 2 == 0:
				optimizer.step()
				optimizer.zero_grad()


			tot_train_loss += loss
			train_correct += (pred.argmax(1) == y).type(
				torch.float).sum().item()

		# --------- Evaluation mode -----------
		with torch.no_grad():
			# set the model in evaluation mode
			model.eval()
			for (x, y) in valid_dl:
				(x, y) = (x.to(device), y.to(device))
				# make the predictions and calculate the validation loss
				pred = model(x)
				tot_valid_loss += loss_function(pred, y)
				# calculate the number of correct predictions
				val_correct += (pred.argmax(1) == y).type(
					torch.float).sum().item()

		# calculate the average training and validation loss
		mean_train_loss = tot_train_loss / training_steps
		mean_valid_loss = tot_valid_loss / valid_steps

		# calculate the training and validation accuracy
		train_correct = train_correct / len(train_ds)
		val_correct = val_correct / len(valid_ds)

		# update our training history
		history["train_loss"].append(mean_train_loss.cpu().detach().numpy())
		history["train_acc"].append(train_correct)
		history["val_loss"].append(mean_valid_loss.cpu().detach().numpy())
		history["val_acc"].append(val_correct)

		# print the model training and validation information
		print("EPOCH: {}/{}".format(e + 1, epochs))
		print("Training loss: {:.6f}, Train accuracy: {:.4f}".format(
			mean_train_loss, train_correct))
		print("Validation loss: {:.6f}, Validation accuracy: {:.4f}".format(
			mean_valid_loss, val_correct))

	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))
 
	return history, model


# Set model to train
trained_model = train(
	model,
 	train_ds=train_ds, valid_ds=valid_ds,
	train_dl=train_dl, valid_dl=valid_dl,
	lr=cfg.LR,
	batch_size=cfg.FEATURE_EXTRACTION_BATCH_SIZE,
	epochs=cfg.EPOCHS,
	device=cfg.DEVICE
)

# Unpack the trained model and history dictionary
history, model = trained_model

# # Create function to visualise history
# visualise(history, output_plot=cfg.WARMUP_PLOT)

# serialize the model to disk
torch.save(model, cfg.WARMUP_MODEL)