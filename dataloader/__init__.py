from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
# from sklearn.model_selection import KFold


class prepare_loader():
    def __init__(self, main_path, train_transform, transform, batch_size, test_batch_size, num_workers,
                 cross_validation=False):
        self.main_path = main_path
        self.train_transform = train_transform
        self.transform = transform
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.cross_validation = cross_validation

        self.train_path = 'train/'
        self.valid_path = 'valid/'
        self.test_path = 'test/'
        #         self.train_dataset=datasets.ImageFolder(root = self.main_path/self.train_path, transform = train_transform)
        #         self.valid_dataset = datasets.ImageFolder(root = self.main_path/self.valid_path, transform = transform)
        #         self.test_dataset = datasets.ImageFolder(root = self.main_path/self.test_path, transform = transform)
        if self.cross_validation:
            self.valid_dataset = datasets.DatasetFolder(root=self.main_path + self.valid_path, transform=self.transform,
                                                        loader=csv_loader, extensions='.csv')
            self.train_dataset = datasets.DatasetFolder(root=self.main_path + self.train_path, transform=self.transform,
                                                        loader=csv_loader, extensions='.csv')
        else:
            self.valid_dataset = datasets.DatasetFolder(root=self.main_path + self.valid_path, transform=self.transform,
                                                        loader=csv_loader, extensions='.csv')
            self.train_dataset = datasets.DatasetFolder(root=self.main_path + self.train_path,
                                                        transform=self.train_transform, loader=csv_loader,
                                                        extensions='.csv')

        self.test_dataset = datasets.DatasetFolder(root=self.main_path + self.test_path, transform=transform,
                                                   loader=csv_loader, extensions='.csv')

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      shuffle=False)

    def dataset_dict(self):
        return {'train': self.train_dataset, 'valid': self.valid_dataset, 'test': self.test_dataset}

    def loader_dict(self):
        return {'train': self.train_loader, 'valid': self.valid_loader, 'test': self.test_loader}


def npy_loader(path):
    sample = torch.from_numpy(np.load(path)).float()
    return sample


def csv_loader(path):
    sample = np.transpose(torch.tensor(pd.read_csv(path, header=None).iloc[:, 1:].to_numpy()).float(), (1, 0))
    # x=torch.tensor(np.expand_dims(pd.read_csv(file).iloc[:,1], axis=0)).float() : only fhr
    return sample

# dataset = datasets.DatasetFolder(
#     root='PATH',
#     loader=npy_loader,
#     extensions=['.npy']
# )