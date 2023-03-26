import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


def get_train_valid_loader(dir, batch_size, augment, seed_value, save):

    ## using CIFAR 100
    trainset = torchvision.datasets.CIFAR100(
        root=dir, train=True, download=save, transform=None) # no transformation, validation set is not transformed
    print ('-- Loading Train & Validation Data --')
    new_train_len = len(trainset) * 0.8
    new_val_len = len(trainset) * 0.2
    train_set, val_set = torch.utils.data.random_split(trainset, [new_train_len, new_val_len], generator=torch.Generator().manual_seed(seed_value)) # set seed here

    print ('Training Data Loaded:', len(train_set))
    print ('Validation Data Loaded', len(val_set))

    mean_val, std_val = calculate_mean_std(train_set)

    if augment == True:

        transform_train = transforms.Compose([
            transforms.Normalize(mean_val, std_val),
            transforms.RandomCrop(32,4),
            transforms.RandomHorizontalFlip(p=0.5), 
            # transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.Normalize(mean_val, std_val),
            # transforms.ToTensor(),
        ])

        # transform train set
        # train_set.dataset.transform = transform_train
        # val_set.dataset.transform = transform_val
        train_set = ApplyTransform(train_set, transform=transform_train)
        val_set = ApplyTransform(val_set, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader


def get_test_loader(dir, batch_size, norm_value): # norm value takes in 

    mean_val = norm_value[0]
    std_val = norm_value[1]
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val),
    ])
    print ('-- Loading Test Data --')
    test_set = torchvision.datasets.CIFAR100(
        root=dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    print ('Test Data Loaded: ', len(test_set))

    return test_loader

"""
Input: trainset
Returns: mean, and standard deviation of a train dataset
"""

def calculate_mean_std(train_set):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set.dataset.transform = transform_train # convert train set to tensor

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=128, shuffle=True, num_workers=2)
    
    mean = 0.0
    var = 0.0
    for i, data in enumerate(train_loader, 0):
        images, _ = data
        batch_samples = images.size(0) # batch size (number of images in the batch)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)

    mean /= len(train_set)
    var /= len(train_set)
    std = torch.sqrt(var)
    
    # convert to tuple rounded to 4dp
    mean_list = mean.tolist()
    mean_val = tuple(np.round(mean_list, 4))
    std_list = std.tolist()
    std_val = tuple(np.round(std_list, 4))


    return mean_val, std_val


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)