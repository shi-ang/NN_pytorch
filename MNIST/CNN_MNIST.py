# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:16:35 2019
CNN model for MNIST dataset
@author: shian
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp
from PIL import Image

print("torch version:",torch.__version__)

class MNIST(Dataset):
    """
    Inherit class Dataset from troch.utils.data
    """
    def __init__(self, root, transform=None):
        """Intialize the MNIST dataset
        
        Arguments:
            - root: root directory of the dataset
        """
        self.images = []
        self.labels = []
        self.root = root
        self.filenames = []
        self.transform = transform
        
        #read file name
        for i in range(10):
            filenames = glob.glob(osp.join(root, str(i), '*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair 
        
        self._preload()
        
        self.len = len(self.filenames)
        
    def _preload(self):
        """
        Preload dataset to memory
        """
        for filename, lable in self.filenames:
            image = Image.open(filename)
            self.images.append(image.copy())
            image.close()
            self.labels.append(lable)
    
    def __getitem__(self, index):                                                                    #"""WHY"""
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
                
            # May use transform function to transform samples
            # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
            # return image and label
        return image, label
        
    def __len__(self):
        return self.len


class NetSeq(nn.Module):
    def __init__(self):
        super(NetSeq, self).__init__()
        
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU())
        
        self.fc_layers = nn.Sequential(
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(50, 10),
                nn.LogSoftmax(dim=1))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        return self.fc_layers(x)



from time import time

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' %checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' %checkpoint_path)

def train(epoch, log_interval = 100):
    model.train()
    iteration = 0
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, label) in enumerate(trainset_loader):
            data, label = data.to(device), label.to(device)
            
            output = model(data)
            loss = F.nll_loss(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        
        end = time()
        print('{:.2f}s'.format(end-start))
        test() # evaluate at the end of epoch

def train_with_checkpoint(epoch, save_internal, log_internal = 100):
    model.train()
    iteration = 0
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, label) in enumerate(trainset_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if iteration % log_internal == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data), len(trainset_loader.dataset),
                        100. * batch_idx / len(trainset_loader), loss.item()))
            if iteration % save_internal == 0 and iteration > 0:
                save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)
            iteration += 1
        
        end = time()
        print('{:.2f}s'.format(end - start))
        save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)
            
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in testset_loader:
            data, label = data.to(device), label.to(device)
            
            output = model(data)
            test_loss = F.nll_loss(output, label)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))


""" 
Avoid setting num_workers > 0 on Windows
If you have to, wrap the main code inside statement: if __name__ == '__main__': 
Reference; https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224
"""

if __name__ == '__main__':
    trainset = MNIST(root='C:/Users/shian/Documents/ECE/Deep Learning Material/mnist_png/training', transform=transforms.ToTensor())
    trainset_loader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 2)
    testset = MNIST(root='C:/Users/shian/Documents/ECE/Deep Learning Material/mnist_png/testing', transform=transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size = 64, shuffle = True, num_workers = 2)
    
    print(len(trainset))
    print(testset.len)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    model = NetSeq().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)

    train_with_checkpoint(5, save_internal=500, log_internal=100)
    
    model = NetSeq().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)
    
    load_checkpoint('mnist-4690.pth', model, optimizer)
    test()





