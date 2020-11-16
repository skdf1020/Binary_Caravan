import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
from dataclasses import dataclass
import torch, os
from pathlib import Path
import numpy as np
import random
from itertools import permutations
from dataclasses import dataclass
from .augmentations import RandAugment, get_augmentation


def get_transform(x):
    transform_dict = {
        # 'NST_transform': NST_transform,
        'NST_transform_1d': NST_transform_1d
    }
    return transform_dict[x]


class NST_transform_1d():
    '''Ver 1: w/o translate, ver 2: w translate '''

    def __init__(self, spec, ver=1, randaugment=False, rand_n=5, rand_m=5, augments_trial=False, augments='Identity',
                 v=0.1):  # version and specify

        self.transform = transforms.Compose([
            # transforms.Lambda(lambda x: torch.tensor(x))
            # transforms.Resize((spec.height, spec.width)), ## 599,9600
            # transforms.Grayscale(spec.gray_channel),
            # transforms.ToTensor(),
            # transforms.Normalize((spec.mean,),(spec.std,))
            ## training set의 mean, standard로 normalization
        ])

        if ver == 1:  # w/o translatea
            self.train_transform = transforms.Compose([
                #    transforms.Lambda(lambda x: torch.tensor(x))
                # transforms.ToTensor(),
                transforms.Lambda(lambda x: permute_nst_1d(x, units=spec.permute_units)),
                # transforms.Lambda(lambda x: torch.where(x>0.5, torch.tensor([0.0]),x))
                # transforms.Normalize((spec.mean,),(spec.std,)),
                # transforms.Lambda(lambda x: x + torch.empty(x.size()).normal_(mean=spec.noise_mean,std=spec.noise_std))
            ])
            if randaugment:
                self.train_transform.transforms.insert(0, RandAugment(rand_n, rand_m))
            if augments_trial:
                self.train_transform.transforms.insert(0, get_augmentation(augments, v))


#### functions

def time_mask(spec, T=100, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]  ##원래 2
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned
        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned


def time_mask_1d(spec, T=100, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]  ##원래 2
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned
        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned


def freq_mask(spec, F=50, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]  # 원래 1, gray image라 바뀐듯.
    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned
        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()
    return cloned


def permute_nst(spec, units=5):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]  ##원래 2
    unit_length = int(len_spectro / units)
    splited_tensor = torch.split(cloned, tuple([unit_length] * (units - 1) + [len_spectro - unit_length * (units - 1)]),
                                 dim=2)
    splited_tensor = next(permutations(splited_tensor, units))
    permuted_tensor = torch.cat(splited_tensor, dim=2)
    return permuted_tensor


def permute_nst_1d(spec, units=5):
    cloned = spec.clone()
    len_spectro = cloned.shape[1]  ##원래 2
    unit_length = int(len_spectro / units)
    splited_tensor = torch.split(cloned, tuple([unit_length] * (units - 1) + [len_spectro - unit_length * (units - 1)]),
                                 dim=1)
    splited_tensor = next(permutations(splited_tensor, units))
    permuted_tensor = torch.cat(splited_tensor, dim=1)
    return permuted_tensor


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)