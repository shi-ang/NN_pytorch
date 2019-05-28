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

import matplotlib.pyplot as plt
import numpy as np
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
    
def imshow(img):
    img = img / 2 + 0.5 #Unnormalize the image
    img = img.numpy()
    """
    In pytorch, the order of dimension is channel*width*height, but
                     in matplotlib, it is width*height*channel
    """
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' %checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' %checkpoint_path)
    
def train_with_checkpoint(epoch, save_internal, log_internal = 1250):
    net.train()
    iteration = 0
    for ep in range(epoch):
        start = time()
        running_loss = 0.0
        for batch_idx, (data, label) in enumerate(trainset_loader):
            data, label = data.to(device), label.to(device)
            
            output = net(data)
            loss = F.cross_entropy(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            #print every x mini-batches (x = log_internal)
            if iteration % log_internal == (log_internal - 1):
                print('Train Epoch: {} [{} * {} = {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx, len(data), batch_idx * len(data), len(trainset_loader.dataset),
                        100. * batch_idx / len(trainset_loader), running_loss / log_internal))
                running_loss = 0.0
            iteration += 1
            
        end = time()
        print('{:.2f}s'.format(end-start))

def test():
    #will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    net.eval()
    test_loss = 0.0
    correct_number = 0
    average_loss = 0.0
    #impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).
    with torch.no_grad(): 
        for data, label in enumerate(testset_loader):
            data, label = data.to(device), label.to(device)
            
            output = net(data)
            test_loss = F.cross_entropy(output, label)
            _, predicted = torch.max(output, 1) #Returns a named tuple (values, indices) indices is index location
            correct_number += (predicted == lables).sum().item()
            
            average_loss += test_loss
            
        average_loss /= len(testset_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                average_loss, correct_number, len(testset_loader.dataset),
                100. * correct_number / len(testset_loader.dataset)))
    
if __name__ == '__main__':
    """
    The output of torchvision datasets are PILImage images of range [0, 1]. 
    We transform them to Tensors of normalized range [-1, 1].
    """
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
    trainset_loader = DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform = transform)
    testset_loader = DataLoader(testset, batch_size = 4, shuffle = True, num_workers = 0)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    data_sample = iter(trainset_loader)
    images, lables = data_sample.next() #One batch has four images when you set the Dataloader
    imshow(torchvision.utils.make_grid(images))

    
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
              
    optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)
    train_with_checkpoint(2, save_internal=500)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    