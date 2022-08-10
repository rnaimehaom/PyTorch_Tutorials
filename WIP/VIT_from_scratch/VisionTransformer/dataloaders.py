import torch
import torchvision as tv

def preprocess_and_transform_CIFAR(img_pth, transforms_list, batch_size_train=100, batch_size_valid=100):
    transforms_list = []
    transform = tv.transforms.Compose(transforms_list)
    
    # Create the training and validation datasets
    train_dataset = tv.datasets.CIFAR10(img_pth, train=True,
                                                download=True, transform=transform)
    
    valid_dataset = tv.datasets.CIFAR10(img_pth, train=False,
                                                 download=True, transform=transform)
    
    # Train data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                          shuffle=True)
    
    # Testing data loader
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_valid,
                                         shuffle=False)
    
    return train_dataloader, valid_dataloader, transforms_list