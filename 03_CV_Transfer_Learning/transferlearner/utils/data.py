from imutils import paths
import numpy as np
import shutil
import os
import json
from torch.utils.data import DataLoader
from torchvision import datasets
import transferlearner.config as cfg

def copy_images(imagePaths, folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

	for path in imagePaths:
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[1]
		labelFolder = os.path.join(folder, label)

		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)   
    
def save_class_labels_to_json(train_dataset, file_name='labels.json'):
    classes = train_dataset.classes
    print(classes)
    dict_class = (dict(list(enumerate(classes))))
    with open(file_name, 'w') as file:
        json.dump(dict_class,file)
    return classes, dict_class     


def get_dataloader(src_dir, custom_transforms, batch_size, random_shuffle=True):
    print(f'[DATASET INFO] creating dataset in {src_dir}')
    dataset = datasets.ImageFolder(
        root = src_dir, 
        transform=custom_transforms
    )
    print('[DATALOADER] creating dataloader ready for use with PyTorch')
    data_loader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=random_shuffle,
        num_workers=os.cpu_count(),
        # See if you have a GPU on your machine, or not
        pin_memory=True if cfg.DEVICE == 'cuda' else False
    )
    
    
    return (dataset, data_loader)