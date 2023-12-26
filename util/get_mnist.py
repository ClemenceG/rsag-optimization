import torch
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
torch.manual_seed(42)

def get_mnist(batch_size=100, num_workers=1, valid_size=0.2):
    train_data = dsets.MNIST(root = './data', train = True, transform = ToTensor(), download = True)
    test_data = dsets.MNIST(root = './data', train = False, transform = ToTensor())
    loaders = get_loaders(train_data, test_data, batch_size=100, num_workers=1, valid_size=0.2)
    return loaders 

def get_train_valid_sampler(train_data, valid_size=0.2):
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler

def get_loaders(train_data, test_data, batch_size=100, num_workers=1, valid_size=0.2):
    train_sampler, valid_sampler = get_train_valid_sampler(train_data, valid_size)
    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(train_data, batch_size=100, num_workers=num_workers, sampler=train_sampler)
    loaders['valid'] = torch.utils.data.DataLoader(train_data, batch_size=100, num_workers=num_workers, sampler=valid_sampler)
    loaders['test'] = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=num_workers)
    return loaders