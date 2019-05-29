#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:04:16 2019

@author: shiang
"""

import torch.nn as nn
import collections
import BasicModule

class AlexNet(BasicModule):
    """
    code from torchvision/models/alexnet.py
    Framework reference <https://arxiv.org/abs/1404.5997>
    """
    def __init__(self, num_class = 2):
        super(AlexNet, self).__init__()
        
        self.mode_name = 'alexnet'
        self.class_number = num_class
        
        self.conv_layers = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2)),
                ('relu1', nn.ReLU(inplace = True)),
                ('Maxpool1', nn.MaxPool2d(kernel_size = 3, stride = 2)),
                ('conv2', nn.Conv2d(64, 192, kernel_size = 5, padding = 2)),
                ('relu2', nn.ReLU(inplace = True)),
                ('Maxpool2', nn.MaxPool2d(kernel_size = 3, stride = 2)),
                ('conv3', nn.Conv2d(192, 384, kernel_size = 3, padding = 1)),
                ('relu3', nn.ReLU(inplace = True)),
                ('conv4', nn.Conv2d(384, 256, kernel_size = 3, padding = 1)),
                ('relu4', nn.ReLU(inplace = True)),
                ('conv5', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
                ('relu5', nn.ReLU(inplace = True)),
                ('Maxpool5', nn.MaxPool2d(kernel_size = 3, stride = 2))
                ]))
    
        self.fc_layers = nn.Sequential(collections.OrderedDict([
                ('dropout1', nn.Dropout()), 
                ('fc1', nn.Linear(256 * 6 *6, 4096)),
                ('relu6', nn.ReLU(inplace = True)),
                ('dropout2', nn.Dropout()),
                ('fc2', nn.Linear(4096, 4096)),
                ('relu7', nn.ReLU(inplace = True)),
                ('fc3', nn.Linear(4096, self.class_number))
                ]))
        
    def forward(self, x):
        x = self.conv_layers(x)
        # Input is a batch, first dimension indicates the batch size
        # second dimension indicates the neuron size of the fc layer
        # Could also be wrote as x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc_layers(x)
        return x