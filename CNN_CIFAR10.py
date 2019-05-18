# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:41:05 2019
CNN on CIFAR10 dataset
@author: shian
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
    trainset_loader = DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform = transform)
    testset_loader = DataLoader(testset, batch_size = 4, shuffle = True, num_workers = 0)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, ))