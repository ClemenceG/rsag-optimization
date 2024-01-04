import torch
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
torch.manual_seed(42)


class DataLoader:
    """
    Abstract class to download data and convert to loader
    """
    def __init__(
            self,
            batch_size=100,
            num_workers=1 
    ):
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.train_data = None
        self.test_data = None
        self.loaders=None

    def get_mnist(self):
        self.train_data = dsets.MNIST(root = './data', train = True, transform = ToTensor(), download = True)
        self.test_data = dsets.MNIST(root = './data', train = False, transform = ToTensor())
        # loaders = DataLoader.get_loaders(train_data, test_data, batch_size=100, num_workers=1, valid_size=0.2)
        return self.train_data, self.test_data 

    def _get_train_valid_sampler(self, valid_size=0.2):
        num_train = len(self.train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler, valid_sampler

    def get_loaders(self, valid_size=0.2):
        if self.train_data is None or self.test_data is None:
            self.get_mnist()
        train_sampler, valid_sampler = self._get_train_valid_sampler(valid_size)
        self.loaders = {}
        self.loaders['train'] = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)
        self.loaders['valid'] = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, sampler=valid_sampler)
        self.loaders['test'] = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.loaders