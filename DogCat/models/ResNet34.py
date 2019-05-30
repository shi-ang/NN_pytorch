# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:16:58 2019

@author: shian
"""

from models.BasicModule import BasicModule 
import torchvision.models as models
import torch.nn as nn

class ResNet34(BasicModule):
    def __init__(self, num_classes = 2):
        super(ResNet34, self).__init__()
        
        self.model_name = 'resnet34'
        
        self.model = models.resnet34(pretrained = False)
        self.model.num_classes = num_classes
        # Using BasicBlock instead of Bottleneck, block.expansion = 1
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    
        