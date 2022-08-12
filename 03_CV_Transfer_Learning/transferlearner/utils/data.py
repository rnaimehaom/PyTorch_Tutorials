from imutils import paths
import numpy as np
import shutil
import os
import json


def get_and_copy_images(img_pths, folder):
    # Check the destination folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Loop over the image paths
    for path in img_pths:
       img_name = path.split(os.path.sep)[-1]
       label = path.split(os.path.sep)[-2] 
       lbl_folder = os.path.join(folder, label)
       
       if not os.path.exists(lbl_folder):
           os.makedirs(lbl_folder)
           
       destination = os.path.join(lbl_folder, img_name)
       shutil.copy(path, destination)
        
           
def create_train_and_val_dirs(dataset_path:str, val_split:float=0.2, train_dir_name:str='images/train', val_dir_name:str='images/valid'):
    print("[INFO] loading image paths...")
    imagePaths = list(paths.list_images(dataset_path))
    np.random.shuffle(imagePaths)

    # generate training and validation paths
    valid_path_len = int(len(imagePaths) * val_split)
    train_path_len = len(imagePaths) - valid_path_len
    trainPaths = imagePaths[:train_path_len]
    valPaths = imagePaths[train_path_len:]

    # copy the training and validation images to their respective
    # directories
    print("[INFO] copying training and validation images...")
    get_and_copy_images(trainPaths, train_dir_name)
    get_and_copy_images(valPaths, val_dir_name)
    
    print("[INFO] copying completed...")  
    
    
def save_class_labels_to_json(train_dataset, file_name='labels.json'):
    classes = train_dataset.classes
    print(classes)
    dict_class = (dict(list(enumerate(classes))))
    with open(file_name, 'w') as file:
        json.dump(dict_class,file)
    return classes, dict_class     