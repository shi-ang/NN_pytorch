# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:46:50 2019

Rewrite the torchvision.models.resnet package

@author: shian
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import collections
import torchvision.models.resnet as resnet

def conv3x3(input_planes, output_planes, stride = 1):
    return nn.Conv2d(input_planes, output_planes, kernel_size = 3, stride = stride, 
                     padding = 1, bias = False)
    
def conv1x1(input_planes, output_planes, stride = 1):
    return nn.Conv2d(input_planes, output_planes, kernel_size = 1, stride = stride, 
                     padding = 0, bias = False)
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, input_planes, internal_planes, stride = 1,  downsample = None):
        super(BasicBlock, self).__init__()
        self.res_path = nn.Sequential(collections.OrderedDict([
                ('conv1', conv3x3(input_planes, internal_planes, stride = stride)),
                ('bn1', nn.BatchNorm2d(internal_planes)),
                ('relu1', nn.ReLU()),
                ('conv2', conv3x3(internal_planes, internal_planes * self.expansion)),
                ('bn2', nn.BatchNorm2d(internal_planes * self.expansion))
                ]))
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        print('Use BasicBlock')
    
    def forward(self, x):
        x = self.res_path(x)
        
        if self.downsample is None:
            shortcut = x
        else:
            shortcut = self.downsample(x)
        
        x = x + shortcut
        x = self.relu2(x)
        return x
    


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, input_planes, internal_planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        
        self.res_path = nn.Sequential(collections.OrderedDict([
                ('conv1', conv1x1(input_planes, internal_planes, stride = stride)),
                ('bn1', nn.BatchNorm2d(internal_planes)),
                ('relu1', nn.ReLU()),
                ('conv2', conv3x3(internal_planes, internal_planes)),
                ('bn2', nn.BatchNorm2d(internal_planes)),
                ('relu2', nn.ReLU()),
                ('conv3', conv1x1(internal_planes, internal_planes * self.expansion)),
                ('bn3', nn.BatchNorm2d(internal_planes * self.expansion))
                ]))
        self.relu3 = nn.ReLU()
        self.downsample = downsample
        print('Use Bottleneck')
        
    def forward(self, x):
        x = self.res_path(x)
        
        if self.downsample is None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
            
        x = x + shortcut
        x = self.relu3(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, blocktype, layers, num_classes = 100, zero_init_residual = False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.pre_layers = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, 
                                    bias = False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU()),
                ('maxpool', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
                ]))
        self.layer1 = self._make_layers(blocktype, 64, layers[0], stride = 1)
        self.layer2 = self._make_layers(blocktype, 128, layers[1], stride = 2)
        self.layer3 = self._make_layers(blocktype, 256, layers[2], stride = 2)
        self.layer4 = self._make_layers(blocktype, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocktype.expansion, num_classes)
        print('Use ResNet')
        
        # Discard the initialization
        
    def _make_layers(self, blocktype, internal_planes, blocks_number, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != internal_planes * blocktype.expansion:
            downsample = nn.Sequential(collections.OrderedDict([
                    ('conv', conv1x1(self.inplanes, internal_planes * blocktype.expansion,
                                     stride = stride)),
                    ('bn', nn.BatchNorm2d(internal_planes * blocktype.expansion))
                    ]))
        
        layers = []
        layers.append(blocktype(self.inplanes, internal_planes, stride = stride, 
                                     downsample = downsample))
        for _ in range(1, blocks_number):
            layers.append(blocktype(self.inplanes, internal_planes))
        """
        Use each element in list layers as the arguments in the Method
        Equals to nn.Sequential(layers[0], layers[1], layer[2], ...)
        """
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
# Compare this class to the library
model = ResNet(BasicBlock, [2, 2, 2, 2]) #resnet18
print(model)

model1 = resnet.resnet18()
print(model1)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    