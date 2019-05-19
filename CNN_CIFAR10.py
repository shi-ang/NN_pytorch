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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import collections
from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(3, 6, 5)),
                ('relu1', nn.ReLU()),
                ('pooling1', nn.MaxPool2d(2)),
                ('conv2', nn.Conv2d(6, 16, 5)),
                ('relu2', nn.ReLU()),
                ('pooling2', nn.MaxPool2d(2))
                ]))
    
        self.fc_layers = nn.Sequential(collections.OrderedDict([
                ('fc1', nn.Linear(16 * 5 * 5, 120)),
                ('relu3', nn.ReLU()),
                ('fc2', nn.Linear(120, 84)),
                ('relu4', nn.ReLU()),
                ('fc3', nn.Linear(84, 10))
                ]))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc_layers(x)
        return x

def train_with_checkpoint(epoch, save_internal, log_internal = 100):
    net.train()
    iteration = 0
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, label) in enumerate(trainset_loader):
            data, label = data.to(device), label.to(device)
            
            output = net(data)
            loss = F.cross_entropy(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % log_internal == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data), len(trainset_loader.dataset),
                        100. * batch_idx / len(trainset_loader), loss.item()))
        
        end = time()
        print('{:.2f}s'.format(end-start))
        
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
    
    print('Number of Traning Sample: %i' %len(trainset))
    print('Number of Train Batch Size: %i' %len(trainset_loader))
    print('Number of Testing Sample: %i' %len(testset))
    print('Number of Test Batch Size: %i' %len(testset_loader))
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    net = Net().to(device)
    print(net)
              
    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
    train_with_checkpoint(5, save_internal=500)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    